#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.exponential import Exponential
from torch.utils.data import DataLoader

from modules.train import input_masking

import numpy as np
import pandas as pd
from tqdm import tqdm
#%%
class UVAE(nn.Module):
    def __init__(self, config, EncodedInfo, device):
        super(DImVAE, self).__init__()
        
        self.config = config
        self.EncodedInfo = EncodedInfo
        self.device = device
        
        self.cont_dim = self.EncodedInfo.num_continuous_features
        self.disc_dim = sum(self.EncodedInfo.num_categories)
        self.p = 2 * self.cont_dim + self.disc_dim + len(self.EncodedInfo.num_categories)
        self.hidden_dim = self.cont_dim + self.disc_dim
        
        """encoder"""
        self.encoder = nn.Sequential(
            nn.Linear(self.p, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, config["latent_dim"] * 2),
        ).to(device)
        
        """decoder"""
        self.delta = torch.arange(0, 1 + config["step"], step=config["step"]).view(1, -1).to(device)
        self.M = self.delta.size(1) - 1
        self.decoder = nn.Sequential(
            nn.Linear(config["latent_dim"], 16),
            nn.ELU(),
            nn.Linear(16, 64),
            nn.ELU(),
            nn.Linear(64, self.cont_dim * (1 + (self.M + 1)) + self.disc_dim),
        ).to(device)
        
        """prior"""
        self.prior = nn.Sequential(
            nn.Linear(self.p, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, config["latent_dim"]), # only mean vector
        ).to(device)
    
    def get_prior(self, conditional_batch):
        mean = self.prior(conditional_batch)
        logvar = (self.config["prior_var"] * torch.ones(mean.shape).to(self.device)).log()
        return mean, logvar
    
    def get_posterior(self, input):
        h = self.encoder(input)
        mean, logvar = torch.split(h, self.config["latent_dim"], dim=1)
        return mean, logvar
    
    def sampling(self, mean, logvar):
        noise = torch.randn(mean.size(0), self.config["latent_dim"]).to(self.device) 
        z = mean + torch.exp(logvar / 2) * noise
        return z
    
    def encode(self, input):
        mean, logvar = self.get_posterior(input)
        z = self.sampling(mean, logvar)
        return z, mean, logvar
    
    def quantile_parameter(self, z):
        h = self.decoder(z)
        
        if self.disc_dim > 0:
            # categorical
            logit = h[:, -self.disc_dim:]
            # continuous
            spline = h[:, :-self.disc_dim]
        else:
            logit = torch.empty(z.size(0), 0).to(self.device)
            spline = h 
        
        
        h = torch.split(spline, 1 + (self.M + 1), dim=1)
        gamma = [h_[:, [0]] for h_ in h]
        beta = [torch.cat([
            torch.zeros_like(gamma[0]),
            nn.ReLU()(h_[:, 1:]) # positive constraint (monotone increasing)
        ], dim=1) for h_ in h]
        beta = [b[:, 1:] - b[:, :-1] for b in beta]
        return gamma, beta, logit
    
    def quantile_function(self, alpha, gamma, beta, j):
        return gamma[j] + (beta[j] * torch.where(
            alpha - self.delta > 0,
            alpha - self.delta,
            torch.zeros(()).to(self.device)
        )).sum(dim=1, keepdims=True)
        
    def _quantile_inverse(self, x, gamma, beta, j):
        delta_ = self.delta.unsqueeze(2).repeat(1, 1, self.M + 1)
        delta_ = torch.where(
            delta_ - self.delta > 0,
            delta_ - self.delta,
            torch.zeros(()).to(self.device))
        mask = gamma[j] + (beta[j] * delta_.unsqueeze(2)).sum(dim=-1).squeeze(0).t()
        mask = torch.where(
            mask <= x, 
            mask, 
            torch.zeros(()).to(self.device)).type(torch.bool).type(torch.float)
        alpha_tilde = x - gamma[j]
        alpha_tilde += (mask * beta[j] * self.delta).sum(dim=1, keepdims=True)
        alpha_tilde /= (mask * beta[j]).sum(dim=1, keepdims=True) + self.config["threshold"]
        alpha_tilde = torch.clip(alpha_tilde, self.config["threshold"], 1) # numerical stability
        return alpha_tilde

    def quantile_inverse(self, x, gamma, beta):
        alpha_tilde_list = []
        for j in range(self.cont_dim):
            alpha_tilde = self._quantile_inverse(x[:, [j]], gamma, beta, j)
            alpha_tilde_list.append(alpha_tilde)
        return alpha_tilde_list
    
    def forward(self, input):
        z, mean, logvar = self.encode(input)
        gamma, beta, logit = self.quantile_parameter(z)
        return z, mean, logvar, gamma, beta, logit
    
    def gumbel_sampling(self, size, eps=1e-20):
        U = torch.rand(size)
        G = (- (U + eps).log() + eps).log()
        return G
    
    def impute(self, train_dataset, seed=0):
        torch.random.manual_seed(seed)
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False)
        
        imputed = []
        for batch in tqdm(train_dataloader, desc="Imputation..."):
            with torch.no_grad():
                batch = batch.to(self.device)
                nan_mask = ~batch.isnan()
                input_batch = input_masking(batch, nan_mask, self)
                
                prior_mean, prior_logvar = self.get_prior(input_batch)
                z = self.sampling(prior_mean, prior_logvar)
                gamma, beta, logit = self.quantile_parameter(z)
                
                samples = []
                # continuous
                for j in range(self.EncodedInfo.num_continuous_features):
                    alphas = torch.rand(batch.size(0), 1).to(self.device)
                    tmp = []
                    for k in range(alphas.shape[1]):
                        tmp.append(self.quantile_function(
                            alphas[:, k].view(batch.size(0), -1), gamma, beta, j)
                        )

                    samples.append(
                        torch.mean(
                            torch.cat(tmp, dim=1), dim=1, keepdim=True
                        )
                    ) ### inverse transform sampling
                # categorical
                st = 0
                for j, dim in enumerate(self.EncodedInfo.num_categories):
                    ed = st + dim
                    out = logit[:, st : ed]
                    G = self.gumbel_sampling(out.shape).to(self.device)
                    _, out = (nn.LogSoftmax(dim=1)(out) + G).max(dim=1) ### Gumbel-Max Sampling
                    samples.append(out.unsqueeze(1))
                    st = ed
                samples = torch.cat(samples, dim=1)
                
                batch[~nan_mask] = samples[~nan_mask]
                imputed.append(batch)
                
        data = torch.cat(imputed, dim=0)
        data = pd.DataFrame(
            data.cpu().numpy(), 
            columns=train_dataset.features)
        
        """un-standardization of synthetic data"""
        for col, scaler in train_dataset.scalers.items():
            data[[col]] = scaler.inverse_transform(data[[col]])
        
        """post-process"""
        data[train_dataset.categorical_features] = data[train_dataset.categorical_features].astype(int)
        data[train_dataset.integer_features] = data[train_dataset.integer_features].round(0).astype(int)
        
        return data
#%%