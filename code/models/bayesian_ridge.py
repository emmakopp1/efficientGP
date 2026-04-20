"""
Bayesian ridge regression for FCS Prediction
"""

# ============================================================================
# IMPORT LIBRAIRIES
# ============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import time

from utils.model_utils import logit, logit_inverse


# ============================================================================
# DEFINE PATHS & LOAD DATA 
# ============================================================================

# Project root directory (3 levels up from this file)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
date = time.strftime("%Y%m%d")

# bayesian ridge output
path_coef = BASE_DIR / "output" / "data" / 'nga_bayesian_ridge' / "nga_significant_coef.txt"
path_coef_chad = BASE_DIR / "output" / "data" / 'tcd_bayesian_ridge' / "tcd_significant_coef.txt"
dir_output_br_prediction = BASE_DIR / "output" / "data" / 'nga_bayesian_ridge'
dir_output_br_prediction_tcd = BASE_DIR / "output" / "data" / 'tcd_bayesian_ridge'

dir_output_br_prediction.mkdir(parents=True, exist_ok=True)
dir_output_br_prediction_tcd.mkdir(parents=True, exist_ok=True)

# input data for nigeria 
path_nigeria_new = BASE_DIR / "data" / "new" / "Nigeria" / "Nigeria-weekly-with-features.csv"

# input data for chad 
path_chad_new = BASE_DIR / "data" / "new" / "Chad" / "Chad-weekly-with-features.csv"


# Read nigerian file filter only the data with 90 days survey
data_nigeria_new = (
    pd.read_csv(path_nigeria_new)
    .query("data_type == 'SURVEY 90 days'")
    )

# Read chad file filter only the data with 90 days survey
data_chad_new = (pd.read_csv(path_chad_new))

# ============================================================================
# SELECT THE FEATURES
# ============================================================================

# Selected features 
'''
From rstudio

  "rainfall_3_months_anomaly_cubic_intp",
  "ndvi_value_cubic_intp",
  "ndvi_anomaly_cubic_intp",
  "log_currency_exchange",
  "log_Area",
  "MPI_intensity",
  "Muslim",
  "log_food_inflation_MPI",
  "log_fatalities_explosions",
  "MPI",
  "n_conflicts_Battles_rolsum_90days",
  "log_fatalities_battles",
  "headline_inflation_cubic_intp",
  "day_num"
'''

X_names = [
    'rainfall_3_months_anomaly_cubic_intp', 
    'ndvi_anomaly_cubic_intp',
    'n_fatalities_Battles_rolsum_90days',
    'n_fatalities_Violence_against_civilians_rolsum_90days',
    'food_inflation_cubic_intp', 
    'currency_exchange_cubic_intp',
    'Ramadan_past90days', 
    'Longitude', 
    'Latitude', 
    'MPI',
    'Muslim', 
    'day_num', 
    ]

X_names_chad = [
    "MPI_intensity",
    "currency_exchange_log",
    "ndvi_value_cubic_intp",
    "food_inflation_cubic_intp",
    "Waterways"
]

Y_names = ['fcs']

# Matrix data for Nigeria 
X = data_nigeria_new[X_names + ['adm1_code','adm1_name','Datetime']]
y = data_nigeria_new[Y_names]
y = logit(y)

# Matrix data for Chad
X_chad = data_chad_new[X_names_chad + ['adm1_code','adm1_name','Datetime']]
y_chad = data_chad_new[Y_names]
y_chad = logit(y_chad)

# ============================================================================
# CREATE DATASET -- NIGERIA 
# ============================================================================

# create manually cv folds according to the spatial cv strategy: each fold has some regions in the test 
# and every other region in the training set (this is less complicated than XGBoost that had also the validation set)
# create indexes
cv_ind = {}
for i in range(0, 6):
    cv_ind[i] = {
        'test': np.arange(len(data_nigeria_new))[(data_nigeria_new.cv_ind == i)],
        'train': np.arange(len(data_nigeria_new))[(data_nigeria_new.cv_ind != i) & (~ pd.isna(data_nigeria_new.cv_ind))]
    }

# create training and test splits according to the cv folds
X_test_dateid = {}

for i in range(0, 6):

    X_test_dateid[i] = (
        X.iloc[cv_ind[i]['test']].copy()
        .pipe(lambda df: df.assign(
            n_fatalities_Battles_rolsum_90days = np.log(1 + df['n_fatalities_Battles_rolsum_90days']),
            n_fatalities_Violence_against_civilians_rolsum_90days=np.log(1 + df['n_fatalities_Violence_against_civilians_rolsum_90days'])
        ))
    )



# Cross validation for train et test
Y_names = ['fcs']

X = data_nigeria_new[X_names]
y = data_nigeria_new[Y_names].squeeze() 
y = logit(y)
X_train = {}
y_train = {}
X_test = {}
y_test = {}

for i in range(0, 6):
    X_train[i] = (
        X.iloc[cv_ind[i]['train']].copy()
        .pipe(lambda df: df.assign(
            n_fatalities_Battles_rolsum_90days=np.log(1 + df['n_fatalities_Battles_rolsum_90days']),
            n_fatalities_Violence_against_civilians_rolsum_90days=np.log(1 + df['n_fatalities_Violence_against_civilians_rolsum_90days'])
        ))
    )
    y_train[i] = (
        y.iloc[cv_ind[i]['train']]
        .copy()
        )
    
    X_test[i] = (
        X.iloc[cv_ind[i]['test']].copy()
        .pipe(lambda df: df.assign(
            n_fatalities_Battles_rolsum_90days=np.log(1 + df['n_fatalities_Battles_rolsum_90days']),
            n_fatalities_Violence_against_civilians_rolsum_90days=np.log(1 + df['n_fatalities_Violence_against_civilians_rolsum_90days'])
        ))
    )
    y_test[i] = (
        y.iloc[cv_ind[i]['test']]
        .copy()
        )

# ============================================================================
# CREATE DATASET -- CHAD 
# ============================================================================

# create manually cv folds according to the spatial cv strategy: each fold has some regions in the test 
# and every other region in the training set (this is less complicated than XGBoost that had also the validation set)
# create indexes
cv_ind_chad = {}
for i in range(0, 6):
    cv_ind_chad[i] = {
        'test': np.arange(len(data_chad_new))[(data_chad_new.cv_ind == i)],
        'train': np.arange(len(data_chad_new))[(data_chad_new.cv_ind != i) & (~ pd.isna(data_chad_new.cv_ind))]
    }

# create training and test splits according to the cv folds
X_test_dateid_chad = {}

for i in range(0, 6):
    X_test_dateid_chad[i] = (
        X_chad.iloc[cv_ind_chad[i]['test']].copy()
    )


# Cross validation for train et test
Y_names = ['fcs']

X_chad = data_chad_new[X_names_chad]
y_chad = data_chad_new[Y_names].squeeze() 
y_chad = logit(y_chad)
X_train_chad = {}
y_train_chad = {}
X_test_chad = {}
y_test_chad = {}

for i in range(0, 6):
    X_train_chad[i] = (
        X_chad.iloc[cv_ind_chad[i]['train']].copy()
    )
    y_train_chad[i] = (
        y_chad.iloc[cv_ind_chad[i]['train']]
        .copy()
        )
    
    X_test_chad[i] = (
        X_chad.iloc[cv_ind_chad[i]['test']].copy()
    )
    y_test_chad[i] = (
        y_chad.iloc[cv_ind_chad[i]['test']]
        .copy()
        )

# ============================================================================
# FIT THE MODELS - NIGERIA 
# ============================================================================

# define numerical columns
numerical_columns = [
    'rainfall_3_months_anomaly_cubic_intp', 
    'ndvi_anomaly_cubic_intp',
    'n_fatalities_Battles_rolsum_90days',
    'n_fatalities_Violence_against_civilians_rolsum_90days',
    'food_inflation_cubic_intp', 
    'currency_exchange_cubic_intp',
    'Longitude', 
    'Latitude', 
    'MPI',
    'Muslim'
    ]

brr_fit = {}
y_pred = {}
R2 = {}
sd_pred = {}
y_pred_logit = {}
var_model = {}

for i in range(0, 6):

    # define the standard scaler
    standardscaler = ColumnTransformer(
        transformers=[
            ("cat", StandardScaler(), numerical_columns),
        ],
        remainder="passthrough"
    )

    # define the pipeline of the one hot encoder and the model
    brr_pipeline= make_pipeline(
        standardscaler,
        linear_model.BayesianRidge(compute_score=True)
    )
    
    brr_fit[i] = brr_pipeline.fit(X_train[i], y_train[i])
    y_pred_logit[i], sd_pred[i] = brr_fit[i].predict(X_test[i], return_std=True)
    y_pred[i] = logit_inverse(y_pred_logit[i])
    R2[i] = brr_fit[i].score(X_train[i], y_train[i])
    var_model[i] = (1/brr_fit[i].named_steps['bayesianridge'].alpha_)**2

# ============================================================================
# FIT THE MODELS - CHAD
# ============================================================================

# define numerical columns
numerical_columns_chad = [
    "MPI_intensity",
    "currency_exchange_log",
    "ndvi_value_cubic_intp",
    "food_inflation_cubic_intp",
    "Waterways"
    ]

brr_fit_chad = {}
y_pred_chad = {}
R2_chad = {}
sd_pred_chad = {}
y_pred_logit_chad = {}
var_model_chad = {}

for i in range(0, 6):
    # define the standard scaler
    standardscaler = ColumnTransformer(
        transformers=[
            ("cat", StandardScaler(), numerical_columns_chad),
        ],
        remainder="passthrough"
    )

    # define the pipeline of the one hot encoder and the model
    brr_pipeline = make_pipeline(
        standardscaler,
        linear_model.BayesianRidge(compute_score=True)
    )
    
    brr_fit_chad[i] = brr_pipeline.fit(X_train_chad[i], y_train_chad[i])
    y_pred_logit_chad[i], sd_pred_chad[i] = brr_fit_chad[i].predict(X_test_chad[i], return_std=True)
    y_pred_chad[i] = logit_inverse(y_pred_logit_chad[i])
    R2_chad[i] = brr_fit_chad[i].score(X_train_chad[i], y_train_chad[i])
    var_model_chad[i] = (1/brr_fit_chad[i].named_steps['bayesianridge'].alpha_)**2


# ============================================================================
# SAVE THE SIGNIFICANT PARAMETERS - NIGERIA
# ============================================================================

for i in range(0, 6):
    model = brr_fit[i].named_steps['bayesianridge']
    # col_transf = brr_fit[i].named_steps['columntransformer']
    upper_coeff = model.coef_ + np.diag(model.sigma_)
    lower_coeff = model.coef_ - np.diag(model.sigma_)
    significant_coef = []
    with open(path_coef, 'a') as f:
        f.write(f"cv fold {i} significant variables:\n")
    for j in range(len(upper_coeff)):
        significant_coef.append(~((upper_coeff[j])>0 & (lower_coeff[j]<0)))
    for j in range(len(significant_coef)):
        # write inside a txt file the significant coefficients of the model:
        if significant_coef[j]:
            with open(path_coef, 'a') as f:
                f.write(f"{X_names[j]} {model.coef_[j]}\n")

    X_test_dateid[i] = X_test_dateid[i].assign(Datetime=pd.to_datetime(X_test_dateid[i]["Datetime"]))


# ============================================================================
# SAVE THE SIGNIFICANT PARAMETERS - CHAD
# ============================================================================

for i in range(0, 6):
    model = brr_fit_chad[i].named_steps['bayesianridge']
    # col_transf = brr_fit[i].named_steps['columntransformer']
    upper_coeff = model.coef_ + np.diag(model.sigma_)
    lower_coeff = model.coef_ - np.diag(model.sigma_)
    significant_coef = []
    with open(path_coef_chad, 'a') as f:
        f.write(f"cv fold {i} significant variables:\n")
    for j in range(len(upper_coeff)):
        significant_coef.append(~((upper_coeff[j])>0 & (lower_coeff[j]<0)))
    for j in range(len(significant_coef)):
        # write inside a txt file the significant coefficients of the model:
        if significant_coef[j]:
            with open(path_coef_chad, 'a') as f:
                f.write(f"{X_names_chad[j]} {model.coef_[j]}\n")

    X_test_dateid_chad[i] = X_test_dateid_chad[i].assign(Datetime=pd.to_datetime(X_test_dateid_chad[i]["Datetime"]))



# ============================================================================
# PLOTS - NIGERIA 
# ============================================================================

# Bayesian Ridge Regression prediction 
for i in range(0,6):

    # Build DataFrame for plotting by region
    df_plot = pd.DataFrame({
        'Datetime': X_test_dateid[i]["Datetime"],
        'y_hat': y_pred[i], 
        'y_true': logit_inverse(y_test[i]),
        'ci_low': logit_inverse(y_pred_logit[i] - sd_pred[i]),    
        'ci_high': logit_inverse(y_pred_logit[i] + sd_pred[i]),
        'y_pred_var': sd_pred[i],
        'var_model': var_model[i],
        'adm1_name': X_test_dateid[i]["adm1_name"]
    })

    # Save predictions for this fold
    output_path = dir_output_br_prediction / f"nga_predictions_fold{i}.csv"
    df_plot.to_csv(output_path, index=False)
    print(f"Nigeria predictions for fold {i} saved to {output_path}")

# ============================================================================
# PLOTS - CHAD 
# ============================================================================

# Bayesian Ridge Regression prediction 
for i in range(0,6):


    # Build DataFrame for plotting by region
    df_plot_chad = pd.DataFrame({
        'Datetime': X_test_dateid_chad[i]["Datetime"],
        'y_hat': y_pred_chad[i], 
        'y_true': logit_inverse(y_test_chad[i]),
        'ci_low': logit_inverse(y_pred_logit_chad[i] - sd_pred_chad[i]),    
        'ci_high': logit_inverse(y_pred_logit_chad[i] + sd_pred_chad[i]),
        'y_pred_var': sd_pred_chad[i],
        'var_model': var_model_chad[i],
        'adm1_name': X_test_dateid_chad[i]["adm1_name"]
    })

    # Save predictions for this fold
    output_path_chad = dir_output_br_prediction_tcd / f"tcd_predictions_fold{i}.csv"
    df_plot_chad.to_csv(output_path_chad, index=False)
    print(f"Chad predictions for fold {i} saved to {output_path_chad}")


