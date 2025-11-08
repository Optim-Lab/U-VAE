# %%
import pandas as pd
import numpy as np
from collections import namedtuple
from modules import metric_fidelity, metric_utility, metric_congeniality

import warnings
warnings.filterwarnings("ignore", "use_inf_as_na")

Metrics = namedtuple(
    "Metrics",
    [
        "SMAPE",
        "RMSE",
        "MAE",
        "PFC",
        "ASMAPE",
        "ARMSE",
        "AMAE",
        "KL",
        "GoF",
        "MMD",
        "WD",
        "CW",
        "alpha_precision", 
        "beta_recall",
        "base_reg", 
        "syn_reg", 
        "base_cls", 
        "syn_cls",
        "syn_cls1",
        "model_selection", 
        "feature_selection",
        "congeniality_bias", 
        "congeniality_mse"
    ]
)
#%%
def evaluate(syndata, train_dataset, test_dataset, config, device):
    
    print("\n1. Imputation Fidelity\n")
    
    print("\n(continuous) SMAPE...")
    SMAPE = metric_fidelity.SMAPE(train_dataset, syndata)
    
    print("\n(continuous) RMSE...")
    RMSE = metric_fidelity.RMSE(train_dataset, syndata)
    
    print("\n(continuous) MAE...")
    MAE = metric_fidelity.MAE(train_dataset, syndata)
    
    print("\n(categorical) Proportion of Falsely Classified (PFC)...")
    PFC = metric_fidelity.PFC(train_dataset, syndata)
    
    print("\n(distributional) KL-Divergence...")
    KL = metric_fidelity.KLDivergence(train_dataset, syndata)
    
    print("\n(distributional) Goodness Of Fit...")
    GoF = metric_fidelity.GoodnessOfFit(train_dataset, syndata)
    
    print("\n(distributional) Maximum Mean Discrepancy...")
    if config["dataset"] == "covtype":
        MMD = metric_fidelity.MaximumMeanDiscrepancy(
            train_dataset, syndata, large=True)
    else:
        MMD = metric_fidelity.MaximumMeanDiscrepancy(train_dataset, syndata)
    
    print("\n(distributional) Wasserstein Distance...")
    if config["dataset"] == "covtype":
        WD = metric_fidelity.WassersteinDistance(
            train_dataset, syndata, large=True)
    else:
        WD = metric_fidelity.WassersteinDistance(train_dataset, syndata)
    
    print("\n(distributional) Cramer-Wold Distance...")
    CW = metric_fidelity.CramerWoldDistance(
        train_dataset, syndata, device)
    
    print("\n(distributional) alpha-precision, beta-recall...")
    alpha_precision, beta_recall = metric_fidelity.naive_alpha_precision_beta_recall(
        train_dataset, syndata)
    
    print("\n2. Machine Learning utility\n")

    print("\nRegression downstream task...")
    base_reg, syn_reg = metric_utility.regression(train_dataset, test_dataset, syndata)
    
    print("\nClassification downstream task...")
    base_cls, syn_cls, syn_cls1, model_selection, feature_selection = metric_utility.classification(
        train_dataset, test_dataset, syndata)
    
    print("\n3. Cogeniality\n")
    congeniality_bias, congeniality_mse = metric_congeniality.congeniality(
        train_dataset, test_dataset, syndata)
    
    return Metrics(
        SMAPE, RMSE, MAE, PFC, SMAPE+PFC, RMSE+PFC, MAE+PFC,
        KL, GoF, MMD, WD, CW, alpha_precision, beta_recall,
        base_reg,  syn_reg, base_cls,  syn_cls, syn_cls1, model_selection, feature_selection,
        congeniality_bias, congeniality_mse
    )
#%%