# Impute Missing Entries with Uncertainty (U-VAE)
This repository is the official implementation of `Impute Missing Entries with Uncertainty' (AAAI, 2026).
> **_NOTE:_** This repository supports [WandB](https://wandb.ai/site) MLOps platform!

## Overview
<img src="DImVAE.png" alt="image" width="700"/>

## Dataset

Download and add the datasets into `data` folder to reproduce our experimental results.

## Reproducibility

### Arguments
- `--dataset`: dataset options (`anuran`, `banknote`, `breast`, `concrete`, `default`, `kings`,  `letter`, `loan`, `redwine`, `shoppers`, `whitewine`)
- `--missing_type`: how to generate missing (`MCAR`, `MAR`, `MNARL`, `MNARQ`)
- `--missing_rate`: missingness rate (default: `0.3`)
- `--M`: the number of multiple imputation (default: `100`)

### Training 
```
python main.py --dataset <dataset> --missing_type <missing_type> --missing_rate <missing_rate> 
```

### Imputation & Evaluation 
> RQ1. Does U-VAE achieve state-of-the-art performance in single imputation tasks?

```
python imputer.py --dataset <dataset> --missing_type <missing_type> --missing_rate <missing_rate> 
```

> RQ2. Can U-VAE support statistically valid multiple imputation by capturing uncertainty in the imputed values?

```
python imputer.py --dataset <dataset> --missing_type <missing_type> --missing_rate <missing_rate> --M <M>
```

> RQ3. How robust is U-VAE to varying missingness rates and patterns in sensitivity analyses?

```
python imputer.py --dataset <dataset> --missing_type <missing_type> --missing_rate <missing_rate> 
```

## Directory and codes

```
.
+-- data
+-- assets 
+-- datasets
|       +-- preprocess.py
|       +-- raw_data.py
+-- modules 
|       +-- evaluation.py
|       +-- evaluation_multiple.py
|       +-- metric_congeniality.py
|       +-- metric_fidelity.py
|       +-- metric_utility.py
|       +-- missing.py
|       +-- model.py
|       +-- train.py
|       +-- utility.py
+-- main.py
+-- imputer.py
+-- U-VAE_supp.pdf
+-- U-VAE.png
+-- README.md
```
