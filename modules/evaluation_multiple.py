# %%
from tqdm import tqdm
import importlib

import pandas as pd
import numpy as np

from collections import namedtuple
#%%
Metrics = namedtuple(
    "Metrics",
    [
        "bias", "bias_percent", "coverage", "interval"
    ],
)

def multiple_evaluate(train_dataset, model, config):
    """target scientific estimand"""
    data = train_dataset.raw_data[train_dataset.continuous_features]
    true = (data > data.mean(axis=0)).astype(float).mean(axis=0)

    est = []
    var = []
    
    for s in tqdm(range(config["M"]), desc="Multiple Imputation..."):
        imputed = model.impute(train_dataset, seed=s)

        data = imputed[train_dataset.continuous_features]
        binary = (data > data.mean(axis=0)).astype(float)
        p = binary.mean(axis=0)
        est.append(p)
        var.append(p * (1. - p) / len(binary))
        
    Q = np.mean(est, axis=0)
    U = np.mean(var, axis=0) + (config["M"] + 1) / config["M"] * np.var(est, axis=0, ddof=1) # total variance = sample + missing + simulation
    
    lower = Q - 1.96 * np.sqrt(U)
    upper = Q + 1.96 * np.sqrt(U)
    
    bias = float(np.abs(Q - true).mean())
    bias_percent = 100 * float((np.abs((Q - true)/Q)).mean())
    coverage = float(((lower < true) & (true < upper)).mean())
    interval = float((upper - lower).mean())
    
    return Metrics(
        bias, bias_percent, coverage, interval
    )