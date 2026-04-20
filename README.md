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
│   │   ├── gaussian_process/         # R entry-point scripts (one per country × model variant)
│   │   │   ├── gp_runner_lib.R       # Core GP framework: data loading, CV, MCMC, diagnostics
│   │   │   ├── TCD_*.R               # Chad model runs
│   │   │   └── NGA_*.R               # Nigeria model runs
│   │   ├── stan/                     # Stan model definitions
│   │   │   ├── GP_helper.stan        # Shared kernel functions (exponential, fBM, spectral decomp.)
│   │   │   ├── GPst_rho1fixed.stan   # Intercept-only spatio-temporal GP
│   │   │   ├── GPstCOV_rho1fixed_normal_prior.stan         # GP with covariates, normal prior
│   │   │   └── GPstCOV_rho1fixed_ridge_centered_prior.stan # GP with covariates, ridge prior
│   │   ├── utils/
│   │   │   ├── data_prep.R           # Feature standardisation (train-set statistics only)
│   │   │   ├── convergence.R         # Rhat / ESS convergence diagnostics
│   │   │   ├── diagnostic.R          # Posterior predictive checks and visualisation
│   │   │   ├── model_utils.py        # Python helpers (logit transforms)
│   │   │   └── mlp_utils.py          # PyTorch Dataset, MLP architecture, training loop
│   │   ├── bayesian_ridge.py         # Bayesian Ridge regression benchmark
│   │   ├── mlp.py                    # MLP with MC Dropout benchmark
│   │   └── xg-boost.py              # XGBoost with bootstrap confidence intervals benchmark
│   └── analysis/
│       ├── visualise_tcd_result.R    # Comparison plots for Chad
│       ├── visualise_nga_result.R    # Comparison plots for Nigeria
│       └── metrics.ipynb             # Performance metrics notebook
├── data/
│   ├── new/
│   │   ├── chad/                     # Weekly time series + static region data for Chad
│   │   └── nigeria/                  # Weekly time series + static region data for Nigeria
│   ├── nigeria_geospatial/           # Shapefiles for administrative boundaries (Nigeria)
│   └── Foini2023/                    # Reference data from Foini et al. 2023
└── output/
    ├── gaussian_process/             # Per-run timestamped GP outputs
    ├── data/                         # Benchmark model predictions and metrics
    └── figure/                       # Figures for the paper
```

---

## Model description

### Spatio-temporal Gaussian Process

The GP prior is defined over regions $s$ and weeks $t$ with a separable covariance:

$$K(s, t, s', t') = \sigma^2 \cdot k_{\text{space}}(s, s';\, \rho_1) \cdot k_{\text{time}}(t, t';\, \rho_2, H) + \sigma_r^2 \, \delta_{s=s'}$$

- **Spatial kernel** $k_{\text{space}}$: exponential (Ornstein–Uhlenbeck) with range $\rho_1$ (fixed by cross-validation).
- **Temporal kernel** $k_{\text{time}}$: fractional Brownian Motion (fBM) kernel with range $\rho_2$ and Hurst exponent $H$.
- **Random effect** $\sigma_r^2$: region-level white noise.

Covariates are modeled by a linear regression. Two prior specifications are provided:
- **Normal prior**: standard shrinkage towards zero.
- **Ridge-centered prior**: additional centering that facilitates feature selection.

All the gaussian process resultats are estimated with MCMC via [CmdStanR](https://mc-stan.org/cmdstanr/) (4 chains, 500 warmup + 1000 sampling iterations by default). An adaptive loop automatically doubles the iteration budget if convergence criteria ($\hat{R} < 1.05$, ESS > 400) are not met.

### Benchmark models

| Model | Implementation | Uncertainty |
|---|---|---|
| Bayesian Ridge | scikit-learn `BayesianRidge` | Posterior predictive variance |
| MLP | PyTorch (3 hidden layers, MC Dropout) | Monte Carlo samples |
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
- `here` ≥ 1.0.1

Install CmdStan via:
```r
cmdstanr::install_cmdstan()
```

### Python

- Python ≥ 3.9
- `torch`, `xgboost`, `scikit-learn`, `pandas`, `numpy`

```bash
pip install torch xgboost scikit-learn pandas numpy
```

---

## Usage

### 1. Gaussian Process model

Each entry-point script in `code/models/gaussian_process/` encodes a specific country × prior combination. To run Chad with covariates and a normal prior in the background:

```bash
nohup Rscript code/models/gaussian_process/TCD_gp_with_features_rho1fixed_normal_prior.R \
  > output/logs/tcd_normal_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Results are saved to a timestamped subdirectory under `output/gaussian_process/`, e.g.:

```
output/gaussian_process/TCD_gp_with_features_normal_prior_20251216_114735/
├── convergence_diagnostics/       # Rhat, trace plots, autocorrelation
├── prior_posterior_diagnostics/   # Posterior coefficient summaries, PPC plots
├── fit_cv*.rds                    # Stan MCMC draws (one per fold)
└── final_cv_summary.csv           # rMSE, MAE, coverage across folds
```

To cross-validate the spatial range $\rho_1$ first:

```bash
nohup Rscript code/models/gaussian_process/TCD_gp_rho1_cross_validation.R \
  > output/logs/tcd_rho1_cv_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 2. Benchmark models

The Python benchmarks consume the preprocessed train/test CSV files written by the GP pipeline. Pass the GP output directory via `--project-path`:

```bash
# MLP
python code/models/mlp.py \
    --country chad \
    --project-path output/gaussian_process/TCD_gp_with_features_ridge_centered_20251216_114735 \
    --output-data output/data/tcd_mlp \
    --num-epochs 200 --lr 0.0001

# XGBoost
python code/models/xg-boost.py \
    --country nigeria \
    --project-path output/gaussian_process/NGA_gp_with_features_normal_centered_20260217_155435 \
    --output-data output/data/nga_xg_boost \
    --n-bootstrap 50

# Bayesian Ridge
python code/models/bayesian_ridge.py \
    --country chad \
    --project-path output/gaussian_process/TCD_gp_with_features_ridge_centered_20251216_114735 \
    --output-data output/data/tcd_bayesian_ridge
```

### 3. Analysis and figures

After all models have run, generate comparison figures with:

```r
source("code/analysis/visualise_tcd_result.R")
source("code/analysis/visualise_nga_result.R")
```

Quantitative metrics are computed in `code/analysis/metrics.ipynb`.

---

## Data

The weekly food security data used in this paper are collected by the [World Food Programme (WFP)](https://www.wfp.org/) and distributed through the [VAM Food Security Monitoring](https://dataviz.vam.wfp.org/) platform. The processed dataset used here is an extension of the one published by [Foini et al. (2023)](https://github.com/pietro-foini/ISI-WFP).

---

## License

This project is licensed under the GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.
