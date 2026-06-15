# EfficientGP: Efficient Gaussian Process Regression for Food Security Forecasting

This repository contains the code and data accompanying the paper:

> **Filling survey gaps in food security monitoring with spatio-temporal additive Gaussian process models**

We provide a Bayesian spatio-temporal Gaussian Process (GP) model for forecasting food insecurity at the sub-national level, benchmarked against Bayesian Ridge regression, Multi-Layer Perceptron (MLP), and XGBoost.

---

## Overview

Food insecurity early warning systems rely on timely, reliable predictions of food consumption at sub-national scale. This project builds on the baseline established by [Foini et al. (2023)](https://www.nature.com/articles/s41598-023-29700-y) and extends it with an efficient GP formulation that captures both spatial and temporal dependencies while incorporating a rich set of covariates (climate, conflict, economic, and demographic indicators).

**Countries covered:** Chad (TCD) and Nigeria (NGA)

**Target variable:** Food Consumption Score (FCS)

---

## Repository structure

```
efficientGP/
├── code/
│   ├── models/
│   │   ├── gaussian_process/
│   │   │   ├── gp_runner_lib.R                          # Core GP framework: data loading, CV, MCMC, diagnostics
│   │   │   ├── gp_runner_nested_cross_validation.R      # Nested CV for rho1 hyperparameter tuning
│   │   │   ├── NGA_gp_rho1_cross_validation.R           # Nigeria: spatial range rho1 cross-validation
│   │   │   ├── NGA_gp_no_features_rho1fixed.R           # Nigeria: GP without covariates
│   │   │   ├── NGA_gp_with_feature_rho1fixed_normal_prior.R  # Nigeria: GP with covariates, normal prior
│   │   │   ├── NGA_gp_with_feature_rho1fixed_ridge_prior.R   # Nigeria: GP with covariates, ridge prior
│   │   │   ├── TCD_gp_rho1_cross_validation.R           # Chad: spatial range rho1 cross-validation
│   │   │   ├── TCD_gp_no_features_rho1fixed.R           # Chad: GP without covariates
│   │   │   ├── TCD_gp_with_features_rho1fixed_normal_prior.R # Chad: GP with covariates, normal prior
│   │   │   └── TCD_gp_with_features_rho1fixed_ridge_prior.R  # Chad: GP with covariates, ridge prior
│   │   ├── stan/
│   │   │   ├── GP_helper.stan                           # Shared kernel functions (OU, fBM, spectral decomp.)
│   │   │   ├── GPst_rho1fixed.stan                      # Intercept-only spatio-temporal GP
│   │   │   ├── GPstCOV_rho1fixed_normal_prior.stan      # GP with covariates, normal prior
│   │   │   └── GPstCOV_rho1fixed_ridge_centered_prior.stan   # GP with covariates, ridge-centered prior
│   │   ├── utils/
│   │   │   ├── data_prep.R        # Feature engineering and standardisation (train-set statistics only)
│   │   │   ├── convergence.R      # Rhat / ESS convergence diagnostics
│   │   │   ├── diagnostic.R       # Posterior predictive checks and coefficient visualisation
│   │   │   ├── model_utils.py     # Python helpers (logit / inverse-logit transforms)
│   │   │   └── mlp_utils.py       # PyTorch Dataset, MLP architecture, training loop, MC Dropout
│   │   ├── bayesian_ridge.py      # Bayesian Ridge regression benchmark
│   │   ├── mlp.py                 # MLP with MC Dropout benchmark
│   │   ├── xg-boost.py            # XGBoost with bootstrap confidence intervals benchmark
│   │   ├── NGA_shap_analysis.py   # SHAP feature importance — Nigeria
│   │   └── TCD_shap_analysis.py   # SHAP feature importance — Chad
│   └── analysis/
│       ├── visualise_nga_result.R      # Comparison plots for Nigeria
│       ├── visualise_tcd_result.R      # Comparison plots for Chad
│       ├── supplementary_material.R    # Figures for the supplementary material
│       ├── metrics.py                  # Quantitative metrics table (RMSE, MAE, coverage)
│       └── coverage_multi_quantile.R   # Coverage at 50/75/90/95% from GP posterior draws
├── data/
│   ├── new/
│   │   ├── nigeria/               # Weekly time series with features + static region data
│   │   └── chad/                  # Weekly time series with features + static region data
│   └── nigeria_geospatial/        # Shapefiles for administrative boundaries (Nigeria)
└── output/
    ├── gaussian_process/          # Per-run timestamped GP outputs (see structure below)
    ├── data/                      # Benchmark model predictions (one CSV per CV fold)
    ├── figure/                    # Figures for the paper
    └── tables/                    # LaTeX metric tables
```

---

## Model description

### Spatio-temporal Gaussian Process

The GP prior is defined over regions $s$ and weeks $t$:

$$f(s, t) = k_{\text{space}}(s, s') \cdot k_{\text{time}}(t, t') + \sigma_r^2 \cdot \delta_{s=s'}$$

- **Spatial kernel** $k_{\text{space}}$: Ornstein–Uhlenbeck (exponential) with range $\rho_1$, fixed by cross-validation.
- **Temporal kernel** $k_{\text{time}}$: fractional Brownian Motion (fBM) with range $\rho_2$ and Hurst exponent $H$.
- **Region random effect** $\sigma_r^2$: white noise term per region.

Covariates enter through a linear term. Two prior specifications are provided:

| Prior | Stan model | Description |
|---|---|---|
| Normal | `GPstCOV_rho1fixed_normal_prior.stan` | Standard shrinkage towards zero |
| Ridge-centered | `GPstCOV_rho1fixed_ridge_centered_prior.stan` | Additional centering for feature selection |

MCMC estimation is performed via [CmdStanR](https://mc-stan.org/cmdstanr/) (4 chains, 500 warmup + 1000 sampling iterations). An adaptive loop automatically doubles the budget if convergence criteria ($\hat{R} < 1.05$, ESS $> 400$) are not met.

### Benchmark models

| Model | Implementation | Uncertainty estimation |
|---|---|---|
| Bayesian Ridge | scikit-learn `BayesianRidge` | Posterior predictive variance |
| MLP | PyTorch — 3 hidden layers, MC Dropout | Monte Carlo Dropout samples |
| XGBoost | xgboost + scikit-learn `GridSearchCV` | Bootstrap confidence intervals |

All benchmarks share the same leave-one-region-out CV folds and the same standardised feature matrices produced by the GP pipeline.

---

## Requirements

### R

- R ≥ 4.4.1
- `cmdstanr` ≥ 0.9.0 (with a working CmdStan installation)
- `tidyverse` ≥ 2.0.0
- `posterior` ≥ 1.6.1
- `bayesplot` ≥ 1.14.0

Install CmdStan via:
```r
cmdstanr::install_cmdstan()
```

### Python

- Python ≥ 3.9
- `torch`, `xgboost`, `scikit-learn`, `pandas`, `numpy`, `shap`

```bash
pip install torch xgboost scikit-learn pandas numpy shap
```

---

## Reproducing the results

The pipeline runs in four sequential steps.

### Step 1 — Cross-validate the spatial range $\rho_1$

Before fitting the full GP, cross-validate the spatial range parameter for each country:

```bash
nohup Rscript code/models/gaussian_process/NGA_gp_rho1_cross_validation.R \
  > output/logs/nga_rho1_cv.log 2>&1 &

nohup Rscript code/models/gaussian_process/TCD_gp_rho1_cross_validation.R \
  > output/logs/tcd_rho1_cv.log 2>&1 &
```

The optimal $\rho_1$ is printed to the log and must be set manually in the corresponding model script before Step 2.

### Step 2 — Fit the GP models

Each script encodes a specific country × prior combination. Results are saved to a timestamped directory under `output/gaussian_process/`.

```bash
# Nigeria
nohup Rscript code/models/gaussian_process/NGA_gp_with_feature_rho1fixed_normal_prior.R \
  > output/logs/nga_gp_normal.log 2>&1 &

nohup Rscript code/models/gaussian_process/NGA_gp_no_features_rho1fixed.R \
  > output/logs/nga_gp_no_features.log 2>&1 &

# Chad
nohup Rscript code/models/gaussian_process/TCD_gp_with_features_rho1fixed_ridge_prior.R \
  > output/logs/tcd_gp_ridge.log 2>&1 &

nohup Rscript code/models/gaussian_process/TCD_gp_no_features_rho1fixed.R \
  > output/logs/tcd_gp_no_features.log 2>&1 &
```

Each run produces a timestamped output directory:

```
output/gaussian_process/NGA_gp_with_features_normal_centered_YYYYMMDD_HHMMSS/
├── convergence_diagnostics/            # Rhat, ESS, trace plots
├── prior_posterior_diagnostics/
│   ├── ppc_summary_all_folds.csv       # Predictions + quantiles for all test observations
│   └── ppc_statistics_by_fold.csv      # Coverage, RMSE, MAE per fold
├── fit_cv{0..5}.rds                    # Stan MCMC draws (one per CV fold)
├── data_train_cv{0..5}.csv            # Training sets (used by benchmark models)
├── data_test_cv{0..5}.csv             # Test sets
└── final_cv_summary.csv               # Aggregate rMSE, MAE, coverage
```

> **Note:** The output directories referenced in `code/analysis/metrics.py`, `visualise_*_result.R`, and `supplementary_material.R` correspond to specific runs used for the paper. Update these paths if you re-run the models.

### Step 3 — Fit the benchmark models

The Python benchmarks consume the `data_train_cv*.csv` / `data_test_cv*.csv` files written by the GP pipeline. Pass the GP output directory via `--project-path` (relative to the project root).

```bash
# MLP
python code/models/mlp.py \
    --country nigeria \
    --project-path output/gaussian_process/NGA_gp_with_features_normal_centered_20260217_155435 \
    --output-data output/data/nga_mlp

python code/models/mlp.py \
    --country chad \
    --project-path output/gaussian_process/TCD_gp_with_features_ridge_centered_20251216_114735 \
    --output-data output/data/tcd_mlp

# XGBoost
python code/models/xg-boost.py \
    --country nigeria \
    --project-path output/gaussian_process/NGA_gp_with_features_normal_centered_20260217_155435 \
    --output-data output/data/nga_xg_boost

python code/models/xg-boost.py \
    --country chad \
    --project-path output/gaussian_process/TCD_gp_with_features_ridge_centered_20251216_114735 \
    --output-data output/data/tcd_xg_boost

# Bayesian Ridge (paths are set inside the script)
python code/models/bayesian_ridge.py
```

### Step 4 — Analysis and figures

```bash
# Metrics table (RMSE, MAE, Coverage 95%)
python code/analysis/metrics.py

# Coverage at 50 / 75 / 90 / 95% (GP models only, from posterior draws)
Rscript code/analysis/coverage_multi_quantile.R

# Comparison figures
Rscript code/analysis/visualise_nga_result.R
Rscript code/analysis/visualise_tcd_result.R

# Supplementary material figures
Rscript code/analysis/supplementary_material.R

# SHAP feature importance
python code/models/NGA_shap_analysis.py
python code/models/TCD_shap_analysis.py
```

---

## Data

The weekly food security data used in this paper are collected by the [World Food Programme (WFP)](https://www.wfp.org/) and distributed through the [VAM Food Security Monitoring](https://dataviz.vam.wfp.org/) platform. The processed dataset used here is an extension of the one published by [Foini et al. (2023)](https://github.com/pietro-foini/ISI-WFP).

Input data files:

| File | Description |
|---|---|
| `data/new/nigeria/nigeria-weekly-with-features.csv` | Weekly FCS observations + covariates, Nigeria |
| `data/new/nigeria/nigeria_shap.csv` | Subset used for SHAP analysis, Nigeria |
| `data/new/chad/chad-weekly-with-features.csv` | Weekly FCS observations + covariates, Chad |
| `data/new/chad/chad-SHAP.csv` | Subset used for SHAP analysis, Chad |
| `data/nigeria_geospatial/` | Shapefiles for administrative boundaries (Nigeria) |

---

## License

This project is licensed under the GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.
