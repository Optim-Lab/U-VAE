""""
Reference:
[1] https://github.com/tigvarts/vaeac/blob/master/data/evaluate_results.py
[2] https://github.com/jsyoon0823/GAIN/blob/master/utils.py
[3] Synthcity: facilitating innovative use cases of synthetic data in different data modalities
- https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/metrics/eval_statistical.py
"""
#%%
from tqdm import tqdm
import numpy as np
import torch

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from geomloss import SamplesLoss
from scipy.stats import entropy, ks_2samp

from modules import utility
#%%
"""Imputation fidelity (continuous): SMAPE, RMSE, MAE"""
def SMAPE(train_dataset, imputed):
    C = train_dataset.num_continuous_features
    original = train_dataset.raw_data.values[:, :C]
    original_ = original[train_dataset.mask[:, :C] == 1]
    
    imputed = imputed.values[:, :C]
    imputed_ = imputed[train_dataset.mask[:, :C] == 1]
    
    smape = np.abs(original_ - imputed_)
    smape /= (np.abs(original_) + np.abs(imputed_)) + 1e-6 # numerical stability
    
    if len(original_) == 0:
        smape = 0
    else:
        smape = smape.mean() 
    
    return smape
#%%
def RMSE(train_dataset, imputed):
    # min-max scaling
    original = train_dataset.raw_data
    original = original[train_dataset.continuous_features]
    imputed = imputed[train_dataset.continuous_features]
    
    scaler = MinMaxScaler().fit(original)
    original = scaler.transform(original)
    imputed = scaler.transform(imputed)
    
    C = train_dataset.num_continuous_features
    original_ = original[train_dataset.mask[:, :C] == 1]
    imputed_ = imputed[train_dataset.mask[:, :C] == 1]
    
    rmse = (original_ - imputed_)**2
    
    if len(original_) == 0:
        rmse = 0
    else:
        rmse = rmse.mean()
        rmse = np.sqrt(rmse)
    
    return rmse
#%%
def MAE(train_dataset, imputed):
    # min-max scaling
    original = train_dataset.raw_data
    original = original[train_dataset.continuous_features]
    imputed = imputed[train_dataset.continuous_features]
    
    scaler = MinMaxScaler().fit(original)
    original = scaler.transform(original)
    imputed = scaler.transform(imputed)
    
    C = train_dataset.num_continuous_features
    original_ = original[train_dataset.mask[:, :C] == 1]
    imputed_ = imputed[train_dataset.mask[:, :C] == 1]
    
    mae = np.abs(original_ - imputed_)
    
    if len(original_) == 0:
        mae = 0
    else:
        mae = mae.mean()
    
    return mae
# %%
"""Imputation fidelity (categorical): Proportion of Falsely Classified (PFC)"""
def PFC(train_dataset, imputed):
    C = train_dataset.num_continuous_features
    original = train_dataset.raw_data.values[:, C:]
    original_ = original[train_dataset.mask[:, C:] == 1]
    
    imputed = imputed.values[:, C:]
    imputed_ = imputed[train_dataset.mask[:, C:] == 1]
    
    if len(original_) == 0:
        pfc = 0
    else:
        pfc = (original_ != imputed_).mean() 
    
    return pfc

#%%
"""Imputation fidelity (distribution): KL, GoF, MMD, WD"""
def KLDivergence(train_dataset, imputed):
    """
    Marginal statistical fidelity: KL-Divergence
    : lower is better
    """
    
    train = train_dataset.raw_data
    num_bins = 10
    
    # get distribution of continuous variables
    cont_freqs = utility.get_frequency(
        train[train_dataset.continuous_features], 
        imputed[train_dataset.continuous_features], 
        n_histogram_bins=num_bins
    )
    
    result = [] 
    for col in train.columns:
        if col in train_dataset.continuous_features:
            gt, syn = cont_freqs[col]
            kl_div = entropy(syn, gt)
        else:
            pmf_p = train[col].value_counts(normalize=True)
            pmf_q = imputed[col].value_counts(normalize=True)
            
            # Ensure that both PMFs cover the same set of categories
            all_categories = pmf_p.index.union(pmf_q.index)
            pmf_p = pmf_p.reindex(all_categories, fill_value=0)
            pmf_q = pmf_q.reindex(all_categories, fill_value=0)
            
            # Avoid division by zero and log(0) by filtering out zero probabilities
            non_zero_mask = (pmf_p > 0) & (pmf_q > 0)
            
            kl_div = np.sum(pmf_q[non_zero_mask] * np.log(pmf_q[non_zero_mask] / pmf_p[non_zero_mask]))
        result.append(kl_div)
    return np.mean(result)
#%%
def GoodnessOfFit(train_dataset, syndata):
    """
    Marginal statistical fidelity: Kolmogorov-Smirnov test & Chi-Squared test
    : lower is better
    """
    
    train = train_dataset.raw_data
    
    result = [] 
    for col in train.columns:
        if col in train_dataset.continuous_features:
            # Compute the Kolmogorov-Smirnov test for goodness of fit.
            statistic, _ = ks_2samp(train[col], syndata[col])
        else:
            pmf_p = train[col].value_counts(normalize=True) # expected
            pmf_q = syndata[col].value_counts(normalize=True) # observed
            
            # Ensure that both PMFs cover the same set of categories
            all_categories = pmf_p.index.union(pmf_q.index)
            pmf_p = pmf_p.reindex(all_categories, fill_value=0)
            pmf_q = pmf_q.reindex(all_categories, fill_value=0)
            
            # Avoid division by zero and log(0) by filtering out zero probabilities
            non_zero_mask = pmf_p > 0
            
            # Compute the Chi-Squared test for goodness of fit.
            statistic = ((pmf_q[non_zero_mask] - pmf_p[non_zero_mask]) ** 2 / pmf_p[non_zero_mask]).sum()
        result.append(statistic)
    return np.mean(result)
#%%
def MaximumMeanDiscrepancy(train_dataset, syndata, large=False):
    """
    Joint statistical fidelity: Maximum Mean Discrepancy (MMD)
    : lower is better
    
    - MMD using RBF (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    """
    
    train = train_dataset.raw_data
    # only continuous
    train = train[train_dataset.continuous_features]
    syndata = syndata[train_dataset.continuous_features]
    
    if large:
        train = train.sample(frac=0.05, random_state=0)
        syndata = syndata.sample(frac=0.05, random_state=0)
    
    scaler = MinMaxScaler().fit(train)
    train_ = scaler.transform(train)
    syndata_ = scaler.transform(syndata)
    
    XX = metrics.pairwise.rbf_kernel(
        train_.reshape(len(train_), -1),
        train_.reshape(len(train_), -1),
    )
    YY = metrics.pairwise.rbf_kernel(
        syndata_.reshape(len(syndata_), -1),
        syndata_.reshape(len(syndata_), -1),
    )
    XY = metrics.pairwise.rbf_kernel(
        train_.reshape(len(train_), -1),
        syndata_.reshape(len(syndata_), -1),
    )
    MMD = XX.mean() + YY.mean() - 2 * XY.mean()
    return MMD
#%%
def WassersteinDistance(train_dataset, syndata, large=False):
    """
    Joint statistical fidelity: Wasserstein Distance
    : lower is better
    """
    
    train = train_dataset.raw_data
    # only continuous
    train = train[train_dataset.continuous_features]
    syndata = syndata[train_dataset.continuous_features]
    
    if large:
        train = train.sample(frac=0.05, random_state=0)
        syndata = syndata.sample(frac=0.05, random_state=0)
    
    train_ = train.values.reshape(len(train), -1)
    syndata_ = syndata.values.reshape(len(syndata), -1)
    
    # assert len(train_) == len(syndata_)

    scaler = MinMaxScaler().fit(train_)
    train_ = scaler.transform(train_)
    syndata_ = scaler.transform(syndata_)
    
    train_ = torch.from_numpy(train_)
    syndata_ = torch.from_numpy(syndata_)
    
    OT_solver = SamplesLoss(loss="sinkhorn")
    """
    Compute WD for 4000 samples and average due to the following error:
    "NameError: name 'generic_logsumexp' is not defined"
    """
    if len(train_) > 4000:
        WD = []
        iter_ = len(train_) // 4000 + 1
        for _ in tqdm(range(iter_), desc="Batch WD..."):
            idx = np.random.choice(range(len(train_)), 4000, replace=False)
            WD.append(OT_solver(train_[idx, :], syndata_[idx, :]).cpu().numpy().item())
        WD = np.mean(WD)
    else:
        WD = OT_solver(train_, syndata_).cpu().numpy().item()
    return WD
#%%
#%%
def phi(s, D):
    return (1 + (4 * s) / (2 * D - 3)) ** (-1 / 2)

def cramer_wold_distance_function(x_batch, x_gen):
    gamma_ = (4 / (3 * x_batch.size(0))) ** (2 / 5)
    
    cw1 = torch.cdist(x_batch, x_batch) ** 2 
    cw2 = torch.cdist(x_gen, x_gen) ** 2 
    cw3 = torch.cdist(x_batch, x_gen) ** 2 
    cw_x = phi(cw1 / (4 * gamma_), D=x_batch.size(1)).sum()
    cw_x += phi(cw2 / (4 * gamma_), D=x_batch.size(1)).sum()
    cw_x += -2 * phi(cw3 / (4 * gamma_), D=x_batch.size(1)).sum()
    cw_x /= (2 * x_batch.size(0) ** 2 * torch.tensor(torch.pi * gamma_).sqrt())
    return cw_x

def CramerWoldDistance(train_dataset, syndata, device):
    """
    Joint statistical fidelity: Cramer-Wold Distance
    : lower is better
    """
    
    train_ = train_dataset.raw_data.copy()
    syndata_ = syndata.copy()
    continuous_features = train_dataset.continuous_features
    categorical_features = train_dataset.categorical_features
    
    # continuous: min-max scaling
    scaler = MinMaxScaler().fit(train_[continuous_features])
    train_[continuous_features] = scaler.transform(train_[continuous_features])
    syndata_[continuous_features] = scaler.transform(syndata_[continuous_features])
    
    # categorical: one-hot encoding
    scaler = OneHotEncoder(handle_unknown='ignore').fit(train_[categorical_features])
    train_ = np.concatenate([
        train_[continuous_features].values,
        scaler.transform(train_[categorical_features]).toarray()
    ], axis=1)
    syndata_ = np.concatenate([
        syndata_[continuous_features].values,
        scaler.transform(syndata_[categorical_features]).toarray()
    ], axis=1)
    train_ = torch.tensor(train_).to(device)
    syndata_ = torch.tensor(syndata_).to(device)
    
    if len(train_) > 10000: # large dataset case
        CWs = []
        for _ in tqdm(range(10), desc="Batch Cramer-Wold Distance..."):
            idx = np.random.choice(range(len(train_)), 2000, replace=False)
            train_small = train_[idx, :]
            syndata_small = syndata_[idx, :]
            cw = cramer_wold_distance_function(train_small, syndata_small)
            CWs.append(cw.cpu().numpy().item())
        cw = np.mean(CWs)
    else:
        cw = cramer_wold_distance_function(train_, syndata_)
        cw = cw.cpu().numpy().item()
    return np.sqrt(cw) # square-root distance
#%%
def naive_alpha_precision_beta_recall(train_dataset, syndata):
    """
    Reference:
    - https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/metrics/eval_statistical.py
    """
    
    train_ = train_dataset.raw_data.copy()
    syndata_ = syndata.copy()
    continuous_features = train_dataset.continuous_features
    categorical_features = train_dataset.categorical_features
    
    # continuous: min-max scaling
    scaler = MinMaxScaler().fit(train_[continuous_features])
    train_[continuous_features] = scaler.transform(train_[continuous_features])
    syndata_[continuous_features] = scaler.transform(syndata_[continuous_features])
    # categorical: one-hot encoding
    scaler = OneHotEncoder(handle_unknown='ignore').fit(train_[categorical_features])
    
    train_ = np.concatenate([
        train_[continuous_features].values,
        scaler.transform(train_[categorical_features]).toarray()
    ], axis=1)
    syndata_ = np.concatenate([
        syndata_[continuous_features].values,
        scaler.transform(syndata_[categorical_features]).toarray()
    ], axis=1)
    
    n_steps = 30
    alphas = np.linspace(0, 1, n_steps) 
    
    emb_center = np.mean(train_, axis=0) # true embedding center
    synth_center = np.mean(syndata_, axis=0) # synthetic embedding center
    
    # L2 distance from true to embedding center
    dist = np.sqrt(((train_ - emb_center) ** 2).sum(axis=1)) 
    # Ball with quantiles of radii 
    # = approximation of the subset that supports a probability mass of alpha
    Radii = np.quantile(dist, alphas) 
    
    # L2 distance from synthetic to embedding center
    synth_to_center = np.sqrt(((syndata_ - emb_center) ** 2).sum(axis=1))
    
    nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(train_)
    real_to_real, _ = nbrs_real.kneighbors(train_) # distance to neighbors

    nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(syndata_)
    # (distance to neighbors, indices of closest synthetic data point to real data)
    real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(train_) 
    
    # Let us find closest real point to any real point, excluding itself (therefore, 1 instead of 0)
    real_to_real = real_to_real[:, 1].squeeze()
    real_to_synth = real_to_synth.squeeze()
    real_to_synth_args = real_to_synth_args.squeeze()
    
    # closest synthetic data points
    # = approximation of true data points using synthetic data points
    real_synth_closest = syndata_[real_to_synth_args] 
    real_synth_closest_d = np.sqrt(((real_synth_closest - synth_center) ** 2).sum(axis=1)) 
    # Ball with quantiles of Radii
    # = approximation of the subset that supports a probability mass of beta
    closest_synth_Radii = np.quantile(real_synth_closest_d, alphas)
    
    alpha_precision_curve = []
    beta_recall_curve = []
    for k in range(len(Radii)):
        alpha_precision = np.mean(
            synth_to_center <= Radii[k]
        )
        beta_recall = np.mean(
            (real_synth_closest_d <= closest_synth_Radii[k]) * (real_to_synth <= real_to_real)
        )
        alpha_precision_curve.append(alpha_precision)
        beta_recall_curve.append(beta_recall)
    
    # Riemann integral
    delta_precision_alpha = 1 - 2 * np.abs(alphas - np.array(alpha_precision_curve)).sum() * (alphas[1] - alphas[0])
    delta_beta_recall = 1 - 2 * np.abs(alphas - np.array(beta_recall_curve)).sum() * (alphas[1] - alphas[0])
    return delta_precision_alpha, delta_beta_recall
#%%
def evaluate(train_dataset, model, M=100, tau=1):
    """target estimand"""
    data = train_dataset.raw_data[train_dataset.continuous_features]
    true = (data > data.mean(axis=0)).astype(float).mean(axis=0)
    # true = pd.DataFrame(train_dataset.raw_data[train_dataset.continuous_features].mean(axis=0))
    
    """multiple imputation"""
    est = []
    var = []
    for s in tqdm(range(M), desc="Multiple Imputation..."):
        imputed = model.impute(train_dataset, tau=tau, seed=s)
        
        data = imputed[train_dataset.continuous_features]
        binary = (data > data.mean(axis=0)).astype(float)
        p = binary.mean(axis=0)
        est.append(p)
        var.append(p * (1. - p) / len(binary))
        # est.append(
        #     pd.DataFrame(imputed[train_dataset.continuous_features].mean(axis=0))
        # )
        # var.append(
        #     pd.DataFrame(imputed[train_dataset.continuous_features].var(axis=0, ddof=1) / len(imputed))
        # )
        
    Q = np.mean(est, axis=0)
    U = np.mean(var, axis=0) + (M + 1) / M * np.var(est, axis=0, ddof=1)
    lower = Q - 1.96 * np.sqrt(U)
    upper = Q + 1.96 * np.sqrt(U)
    
    bias = float(np.abs(Q - true).mean())
    coverage = float(((lower < true) & (true < upper)).mean())
    interval = float((upper - lower).mean())
    
    return Metrics(
        bias, coverage, interval
    )