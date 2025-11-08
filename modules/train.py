#%%
import numpy as np
from tqdm import tqdm
import wandb

import torch
from torch import nn
import torch.nn.functional as F
#%%
def continuous_CRPS(model, x_batch, alpha_tilde_list, gamma, beta, j):
    term = (1 - model.delta.pow(3)) / 3 - model.delta - torch.maximum(alpha_tilde_list[j], model.delta).pow(2)
    term += 2 * torch.maximum(alpha_tilde_list[j], model.delta) * model.delta
    crps = (2 * alpha_tilde_list[j]) * x_batch[:, [j]]
    crps += (1 - 2 * alpha_tilde_list[j]) * gamma[j]
    crps += (beta[j] * term).sum(dim=1, keepdims=True)
    crps *= 0.5
    return crps

def input_masking(batch, mask, model):
    cont_dim = model.EncodedInfo.num_continuous_features
    masked_batch = []

    """indicator method"""
    # continuous
    batch_ = batch.clone()
    batch_[~mask] = 0.
    masked_batch.extend(
        [batch_[:, :cont_dim], (~mask).float()[:, :cont_dim]]
    )
    # categorical
    batch_ = batch.clone()
    batch_[~mask] = torch.nan
    for j, dim in enumerate(model.EncodedInfo.num_categories):
        masked_batch.append(
            F.one_hot(
                batch_[:, cont_dim + j].nan_to_num(dim).long(),
                num_classes=dim+1
            ) # NaN = additional class label
        )
    masked_batch = torch.cat(masked_batch, dim=1)
    return masked_batch

def output_masking(batch, mask, model):
    cont_dim = model.EncodedInfo.num_continuous_features
    masked_batch = []

    # continuous
    batch_ = batch.clone()
    batch_[~mask] = 0.
    masked_batch.append(
        batch_[:, :cont_dim]
    )
    # categorical
    batch_ = batch.clone()
    batch_[~mask] = torch.nan
    for j, dim in enumerate(model.EncodedInfo.num_categories):
        masked_batch.append(
            F.one_hot(
                batch_[:, cont_dim + j].nan_to_num(dim).long(),
                num_classes=dim+1
            )[:, :-1] # remove NaN index column
        )
    masked_batch = torch.cat(masked_batch, dim=1)
    return masked_batch

#%%
def train_function(model, train_dataloader, config, optimizer, device):
    
    for epoch in range(config["epochs"]):
        logs = {
            'loss': [], 
            'recon': [],
            'KL': [],
        }
        # for debugging
        logs['activated'] = []
        
        for x_batch in tqdm(iter(train_dataloader), desc="inner loop"):
            x_batch = x_batch.to(device)
            
            # m = 1: NOT masked
            mask = torch.rand_like(x_batch) > torch.rand(x_batch.size(0), 1).to(device)
            # r = 1: NOT missing
            nan_mask = ~x_batch.isnan()
            
            input_mask = nan_mask
            conditional_mask = nan_mask & mask
            output_mask = nan_mask & (~mask)
            
            input_batch = input_masking(x_batch, input_mask, model)
            conditional_batch = input_masking(x_batch, conditional_mask, model)
            output_batch = output_masking(x_batch, output_mask, model)
            
            optimizer.zero_grad()
            
            z, mean, logvar, gamma, beta, logit = model(input_batch)
            prior_mean, prior_logvar = model.get_prior(conditional_batch)
            
            loss_ = []
            
            """1. Reconstruction loss: CRPS"""
            alpha_tilde_list = model.quantile_inverse(output_batch, gamma, beta)
            recon = 0
            for j in range(model.EncodedInfo.num_continuous_features):
                crps = continuous_CRPS(model, output_batch, alpha_tilde_list, gamma, beta, j)
                crps = crps[output_mask[:, j]].sum() / x_batch.size(0)
                if not crps.isnan():
                    recon += crps
            st = 0
            cont_dim = model.EncodedInfo.num_continuous_features
            for j, dim in enumerate(model.EncodedInfo.num_categories):
                ed = st + dim
                _, targets = output_batch[:, cont_dim + st : cont_dim + ed].max(dim=1)
                out = logit[:, st : ed]
                CE = nn.CrossEntropyLoss()(
                    out[output_mask[:, j]], 
                    targets[output_mask[:, j]]
                )
                if not CE.isnan():
                    recon += CE
                st = ed
            loss_.append(('recon', recon))
            
            """2. KL-Divergence"""
            KL = (torch.pow(mean - prior_mean, 2) / prior_logvar.exp()).sum(dim=1)
            KL += prior_logvar.sum(dim=1) - logvar.sum(dim=1)
            KL += (logvar.exp() / prior_logvar.exp()).sum(dim=1)
            KL -= config["latent_dim"]
            KL *= 0.5
            KL = 1*KL.mean()
            loss_.append(('KL', KL))
            
            """3. ELBO"""
            loss = recon + config["beta"] * KL 
            loss_.append(('loss', loss))
            
            var_ = torch.exp(logvar) < 0.1
            loss_.append(('activated', var_.float().mean()))
            
            loss.backward()
            optimizer.step()
                
            for x, y in loss_:
                logs[x] = logs.get(x) + [y.item()]
        
        print_input = f"Epoch [{epoch+1:03d}/{config['epochs']}]"
        print_input += "".join(
            [", {}: {:.4f}".format(x, np.mean(y)) for x, y in logs.items()]
        )
        print(print_input)
        wandb.log({x : np.mean(y) for x, y in logs.items()})
        
    return
#%%