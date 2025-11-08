"""
Reference:
[1] Reimagining Synthetic Tabular Data Generation through Data-Centric AI: A Comprehensive Benchmark
- https://github.com/HLasse/data-centric-synthetic-data
"""
#%%
import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from scipy.stats import spearmanr
#%%
def regression(train_dataset, test_dataset, imputed):
    train = train_dataset.raw_data.copy()
    test = test_dataset.raw_data.copy()
    imputed = imputed.copy()
    
    """Baseline"""
    print(f"\n(Baseline) Regression: SMAPE...")
    result = []
    for col in train_dataset.continuous_features:
        covariates = [x for x in train.columns if x not in [col]]
        
        regr = RandomForestRegressor(random_state=0, n_jobs=-1)
        regr.fit(train[covariates], train[col])
        pred = regr.predict(test[covariates])
        true = np.array(test[col])
        
        smape = np.abs(true - pred)
        smape /= (np.abs(true) + np.abs(pred)) + 1e-6 # numerical stability
        smape = smape.mean()
        
        result.append((col, smape))
        print("[{}] SMAPE: {:.3f}".format(col, smape))
    base_reg = np.mean([x[1] for x in result])
    
    """Synthetic"""
    print(f"\n(Synthetic) Regression: SMAPE...")
    result = []
    for col in train_dataset.continuous_features:
        covariates = [x for x in imputed.columns if x not in [col]]
        
        regr = RandomForestRegressor(random_state=0, n_jobs=-1)
        regr.fit(imputed[covariates], imputed[col])
        pred = regr.predict(test[covariates])
        true = np.array(test[col])
        
        smape = np.abs(true - pred)
        smape /= (np.abs(true) + np.abs(pred)) + 1e-6 # numerical stability
        smape = smape.mean()
        
        result.append((col, smape))
        print("[{}] SMAPE: {:.3f}".format(col, smape))
    syn_reg = np.mean([x[1] for x in result])
    
    return base_reg, syn_reg
#%%
def classification(train_dataset, test_dataset, imputed):
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
    
    """Baseline"""
    performance = []
    performance1 = []
    feature_importance = []
    print(f"(Baseline) Classification: Accuracy...")
    for name, clf in [
        ('logit', LogisticRegression(tol=0.001, random_state=42, n_jobs=-1, max_iter=1000)),
        ('KNN', KNeighborsClassifier(n_jobs=-1)),
        ('RBF-SVM', SVC(random_state=42)),
        ('RandomForest', RandomForestClassifier(random_state=42, n_jobs=-1)),
        ('GradBoost', GradientBoostingClassifier(random_state=42)),
        ('AdaBoost', AdaBoostClassifier(random_state=42)),
    ]:
        clf.fit(train_[:, :-1], train_[:, -1])
        pred = clf.predict(test_[:, :-1])
        acc = accuracy_score(test_[:, -1], pred)
        f1 = f1_score(test_[:, -1], pred, average='micro')
        
        if name in ["RandomForest", "GradBoost", "AdaBoost"]: 
            feature_importance.append(clf.feature_importances_)
        print(f"[{name}] ACC: {acc:.3f}")
        print(f"[{name}] F1: {f1:.3f}")
        
        performance.append(acc)
        performance1.append(f1)

    base_performance = performance
    base_cls = np.mean(performance)
    base_cls1 = np.mean(performance1)
    base_feature_importance = feature_importance
    
    """Synthetic"""
    if len(np.unique(imputed_[:, -1])) == 0:
        return (
            base_cls, 0., 0., 0., 0.
        )
    else:
        performance = []
        performance1 = []
        feature_importance = []
        print(f"(Synthetic) Classification: Accuracy...")
        for name, clf in [
            ('logit', LogisticRegression(tol=0.001, random_state=42, n_jobs=-1, max_iter=1000)),
            ('KNN', KNeighborsClassifier(n_jobs=-1)),
            ('RBF-SVM', SVC(random_state=42)),
            ('RandomForest', RandomForestClassifier(random_state=42, n_jobs=-1)),
            ('GradBoost', GradientBoostingClassifier(random_state=42)),
            ('AdaBoost', AdaBoostClassifier(random_state=42)),
        ]:
            clf.fit(imputed_[:, :-1], imputed_[:, -1])
            pred = clf.predict(test_[:, :-1])
            acc = accuracy_score(test_[:, -1], pred)
            f1 = f1_score(test_[:, -1], pred, average='micro')
            if name in ["RandomForest", "GradBoost", "AdaBoost"]: 
                feature_importance.append(clf.feature_importances_)
            print(f"[{name}] ACC: {acc:.3f}")
            print(f"[{name}] F1: {f1:.3f}")
            
            performance.append(acc)
            performance1.append(f1)
                
        syn_cls = np.mean(performance)
        syn_cls1 = np.mean(performance1)
        
        model_selection = spearmanr(base_performance, performance).statistic
        feature_selection = []
        for f1, f2 in zip(base_feature_importance, feature_importance):
            feature_selection.append(spearmanr(f1, f2).statistic)
        feature_selection = np.mean(feature_selection)
        
        return (
            base_cls, syn_cls, syn_cls1, model_selection, feature_selection
        )
#%%