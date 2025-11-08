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
        "RMSE",
        "WD",
        "base_reg", 
        "syn_reg", 
        "base_cls", 
        "syn_cls",
        "syn_cls1",
        "feature_selection",
        "congeniality_bias", 
        "congeniality_mse"
    ]
)
#%%
def evaluate(syndata, train_dataset, test_dataset, config, device):
    
    print("\n1. Imputation Fidelity\n")
    
    print("\n(continuous) RMSE...")
    RMSE = metric_fidelity.RMSE(train_dataset, syndata)
    
    print("\n(distributional) Wasserstein Distance...")
    if config["dataset"] == "covtype":
        WD = metric_fidelity.WassersteinDistance(
            train_dataset, syndata, large=True)
    else:
        WD = metric_fidelity.WassersteinDistance(train_dataset, syndata)
    
    print("\n2. Machine Learning utility\n")

    print("\nRegression downstream task...")
    base_reg, syn_reg = metric_utility.regression(train_dataset, test_dataset, syndata)
    
    print("\nClassification downstream task...")
    base_cls, syn_cls, syn_cls1, _, feature_selection = metric_utility.classification(
        train_dataset, test_dataset, syndata)
    
    print("\n3. Cogeniality\n")
    congeniality_bias, congeniality_mse = metric_congeniality.congeniality(
        train_dataset, test_dataset, syndata)
    
    return Metrics(
        RMSE, WD,
        base_reg,  syn_reg, base_cls,  syn_cls, syn_cls1, feature_selection,
        congeniality_bias, congeniality_mse
    )
#%%