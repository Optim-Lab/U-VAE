#%%
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
#%%
def congeniality(train_dataset, test_dataset, imputed):
    ### pre-processing
    continuous = train_dataset.continuous_features
    categorical = [x for x in train_dataset.categorical_features if x != train_dataset.ClfTarget]
    target = train_dataset.ClfTarget
    
    train_ = train_dataset.raw_data.copy()
    test_ = test_dataset.raw_data.copy()
    imputed_ = imputed.copy()
    
    # continuous: standardization
    scaler = StandardScaler().fit(train_[continuous])
    train_[continuous] = scaler.transform(train_[continuous])
    test_[continuous] = scaler.transform(test_[continuous])
    imputed_[continuous] = scaler.transform(imputed_[continuous])

    # categorical: one-hot encoding
    if len(categorical):
        scaler = OneHotEncoder(drop="first", handle_unknown='ignore').fit(train_[categorical])
        train_ = np.concatenate([
            train_[continuous].values,
            scaler.transform(train_[categorical]).toarray(),
            train_[[target]].values
        ], axis=1)
        test_ = np.concatenate([
            test_[continuous].values,
            scaler.transform(test_[categorical]).toarray(),
            test_[[target]].values
        ], axis=1)
        imputed_ = np.concatenate([
            imputed_[continuous].values,
            scaler.transform(imputed_[categorical]).toarray(),
            imputed_[[target]].values
        ], axis=1)
    else:
        train_ = train_.values
        test_ = test_.values
        imputed_ = imputed_.values

    # train logisitic regression
    model_complete = LogisticRegression(
        tol=1e-4, 
        random_state=42, 
        max_iter=1000
    ).fit(train_[:, :-1], train_[:, -1])
    model_imputed = LogisticRegression(
        tol=1e-4, 
        random_state=42, 
        max_iter=1000
    ).fit(imputed_[:, :-1], imputed_[:, -1])

    if model_complete.coef_.shape == model_imputed.coef_.shape:
        w = model_complete.coef_.flatten()
        w_tilde = model_imputed.coef_.flatten()
    else:
        min_rows = min(
            model_complete.coef_.shape[0], 
            model_imputed.coef_.shape[0])
        min_cols = min(
            model_complete.coef_.shape[1], 
            model_imputed.coef_.shape[1])
        w = model_complete.coef_[:min_rows, :min_cols].flatten()
        w_tilde = model_imputed.coef_[:min_rows, :min_cols].flatten()


    # Bias (L1), MSE (L2)
    congeniality_bias = np.abs(w - w_tilde).mean()
    congeniality_mse = np.square(w - w_tilde).mean()

    return (congeniality_bias, congeniality_mse)
