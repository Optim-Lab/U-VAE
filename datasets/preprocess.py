#%%
import pandas as pd
import numpy as np
from collections import namedtuple
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from modules.missing import generate_mask
from datasets.raw_data import load_raw_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EncodedInfo = namedtuple(
    'EncodedInfo', 
    ['num_features', 'num_continuous_features', 'num_categories'])
#%%
class CustomDataset(Dataset):
    def __init__(
        self, 
        config, 
        scalers=None,
        train=True):
        
        self.config = config
        self.train = train
        data, continuous_features, categorical_features, integer_features, ClfTarget = load_raw_data(config["dataset"])
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.integer_features = integer_features
        self.ClfTarget = ClfTarget
        
        self.features = self.continuous_features + self.categorical_features
        self.col_2_idx = {col : i for i, col in enumerate(data[self.features].columns.to_list())}
        self.num_continuous_features = len(self.continuous_features)
        
        # encoding categorical dataset
        data[self.categorical_features] = data[self.categorical_features].apply(
            lambda col: col.astype('category').cat.codes)
        self.num_categories = data[self.categorical_features].nunique(axis=0).to_list()

        data = data[self.features] # select features for training
        train_data, test_data = train_test_split(
            data, test_size=config["test_size"], random_state=config["seed"])
        
        data = train_data if train else test_data
        data = data.reset_index(drop=True)
        self.raw_data = train_data if train else test_data
        
        if train:
            if config["missing_type"] != "None":
                self.mask = generate_mask(
                    torch.from_numpy(data.values).float(), 
                    config["missing_rate"], 
                    config["missing_type"],
                    seed=config["seed"])
                data.mask(self.mask.astype(bool), np.nan, inplace=True)
        
        self.scalers = {} if train else scalers
        transformed = []
        for continuous_feature in tqdm(self.continuous_features, desc="Tranform Continuous Features..."):
            transformed.append(self.transform_continuous(data, continuous_feature))
        
        self.data = np.concatenate(
            transformed + [data[self.categorical_features].values], axis=1
        )
        
        self.EncodedInfo = EncodedInfo(
            len(self.features), self.num_continuous_features, self.num_categories)
        
    def transform_continuous(self, data, col):
        nan_value = data[[col]].to_numpy().astype(float)
        nan_mask = np.isnan(nan_value)
        feature = nan_value[~nan_mask].reshape(-1, 1)
        
        if self.train:
            scaler = StandardScaler().fit(feature)
            self.scalers[col] = scaler
        else:
            scaler = self.scalers[col]
            
        nan_value[~nan_mask] = scaler.transform(feature)[:, 0]
        return nan_value
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])
#%%
