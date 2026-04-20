#!/usr/bin/env python3
"""
XGBoost Nested Cross-Validation Runner
=======================================

Purpose
-------
Predict the Food Consumption Score (FCS) at the administrative level (adm1) for
different countries using an XGBoost model trained with nested spatial
cross-validation.

Pipeline overview
-----------------
This script follows the same structure as mlp.py:

1. **Input data**: CSV files produced by the Gaussian Process pipeline
   (data_train_cvX.csv / data_test_cvX.csv for X in {0..5}).

2. **Nested cross-validation**
   - Outer loop (6 folds): separates train+val from test at each fold.
   - Inner loop (5 splits via get_cv_splits): used by GridSearchCV for
     hyperparameter tuning while respecting the spatial structure of the data
     (regions within the same fold are never mixed across train and val).

3. **Hyperparameter selection**
   GridSearchCV sweeps over PARAM_GRID (n_estimators, max_depth, learning_rate,
   subsample). The best model is chosen by a criterion that jointly minimises the
   train/val gap (min_diff=True) and the validation error.

4. **Bootstrap (n=50 by default)**
   The selected model is retrained n_bootstrap times on resamples of the
   train+val set (with replacement). Bootstrap predictions are used to build
   95% confidence intervals (2.5th–97.5th percentile) on the test set.

5. **Outputs**
   - predictions_cvX.csv: per-fold predictions (y_hat, y_true, ci_low, ci_high,
     rMSE per region, Datetime, adm1_name).
   - <country>_all_predictions.csv: concatenation of all 6 folds.
   - <country>_summary.json: mean ± std RMSE and best hyperparameters per fold.
   - grid_search CSV: full GridSearchCV results for each outer fold.

Key functions
-------------
- train_single_model               : fits one XGBRegressor and returns metrics.
- get_tuning_best_results          : selects the best hyperparameter set from
                                     GridSearchCV results.
- get_model_importance             : computes and normalises feature importances
                                     (total_gain).
- train_model_bootstrap_statial_cv : orchestrates GridSearchCV + spatial bootstrap.
- get_cv_splits                    : builds inner train/val splits for a given
                                     outer fold (identical to mlp.py).
- prepare_features                 : extracts the feature matrix (converts Datetime
                                     to a Unix integer timestamp).
- main                             : CLI argument parsing and main training loop.

Usage
-----
Train on Chad with prediction saving:

    python xg-boost.py \\
        --country chad \\
        --project-path output/gaussian_process/TCD_gp_with_features_ridge_centered_20251216_114735 \\
        --output-data output/data/tcd_xg_boost

Train on Nigeria:

    python xg-boost.py \\
        --country nigeria \\
        --project-path output/gaussian_process/NGA_gp_with_features_normal_centered_20260217_155435/ \\
        --output-data output/data/nga_xg_boost

CLI arguments
-------------
--country        Country name (e.g. chad, nigeria)            [required]
--project-path   Directory containing the GP CSV files        [required]
--output-data    Directory to save results                    [optional]
--cv-folds       Comma-separated fold indices to run          [default: 0,1,2,3,4,5]
--n-bootstrap    Number of bootstrap iterations               [default: 50]
--seed           Random seed for reproducibility              [default: 42]
"""

import argparse
import json
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV

warnings.simplefilter(action="ignore")


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def train_single_model(
    train: pd.DataFrame,
    test: Optional[pd.DataFrame],
    independent_variables: List[str],
    target_variable: str,
    hyperparameters: Dict[str, int],
):
    model = xgb.XGBRegressor(
        verbosity=0, **hyperparameters, n_jobs=2, objective="reg:logistic"
    )
    model.fit(train[independent_variables], train[target_variable])

    y_predict_train = model.predict(train[independent_variables])
    r2_train = r2_score(y_pred=y_predict_train, y_true=train[target_variable])
    mae_train = mean_absolute_error(y_pred=y_predict_train, y_true=train[target_variable])

    if test is not None:
        y_predict_test = model.predict(test[independent_variables])
        r2_test = r2_score(y_pred=y_predict_test, y_true=test[target_variable])
        mae_test = mean_absolute_error(y_pred=y_predict_test, y_true=test[target_variable])
        results = {"r2_test": r2_test, "mae_test": mae_test, "r2_train": r2_train, "mae_train": mae_train}
    else:
        results = {"r2_train": r2_train, "mae_train": mae_train}

    return model, results


def get_tuning_best_results(
    grid: GridSearchCV, min_diff: bool, lambda_parameter: Optional[float] = 1
) -> pd.DataFrame:
    grid_search_full_results = pd.DataFrame(grid.cv_results_)
    grid_search_full_results["r2_diff"] = (
        grid_search_full_results["mean_train_r2"] - grid_search_full_results["mean_test_r2"]
    )

    if min_diff:
        grid_search_full_results["elaborated_function"] = lambda_parameter * (
            grid_search_full_results["mean_train_r2"] - grid_search_full_results["mean_test_r2"]
        ) + (1 - lambda_parameter) * (1.0 - grid_search_full_results["mean_test_r2"])

        r2_min_diff_vec = [
            row["elaborated_function"]
            for _, row in grid_search_full_results.iterrows()
            if row["r2_diff"] >= 0
        ]
        if r2_min_diff_vec:
            r2_min_diff = min(r2_min_diff_vec)
            results_best_model_split = grid_search_full_results.loc[
                grid_search_full_results["elaborated_function"] == r2_min_diff
            ].iloc[0]
        else:
            r2_min_diff = max(grid_search_full_results["r2_diff"])
            results_best_model_split = grid_search_full_results.loc[
                grid_search_full_results["r2_diff"] == r2_min_diff
            ].iloc[0]
    else:
        rmse_min = min(-row["mean_test_neg_root_mean_squared_error"] for _, row in grid_search_full_results.iterrows())
        results_best_model_split = grid_search_full_results.loc[
            grid_search_full_results["mean_test_neg_root_mean_squared_error"] == -rmse_min
        ].iloc[0]

    return results_best_model_split


def get_model_importance(model, boot_id: int):
    importance = model.get_booster().get_score(importance_type="total_gain")
    imp = pd.DataFrame.from_dict(importance, orient="index")
    imp.reset_index(level=0, inplace=True)
    imp.columns = ["feature", "value"]
    imp["value"] = imp["value"] / imp["value"].sum() * 100
    imp = imp.sort_values(by=["value"], ascending=False).reset_index(drop=True)
    imp["iteration"] = boot_id
    return imp


def train_model_bootstrap_statial_cv(
    df_train_and_val: pd.DataFrame,
    independent_variables: Union[list, set],
    target_variable: str,
    last: bool,
    grid_search_dict: Dict,
    config_id: Union[int, str],
    df_test: Optional[Union[pd.DataFrame, list]] = None,
    n_bootstrap: int = 100,
    min_diff: bool = True,
    importance: bool = True,
    list_cv: List[tuple] = None,
    output_dir: Optional[str] = None,
):
    """
    Performs a grid search and trains n_bootstrap models by re-sampling the provided dataset.
    Uses spatial CV splits provided via list_cv.

    Args:
        df_train_and_val: dataframe with training and validation set
        independent_variables: list of explanatory variables
        target_variable: name of the dependent variable
        last: boolean variable which indicates if the previous recorded score
         of the dependent variable is used as explanatory variable
        grid_search_dict: hyperparameters over the grid search is performed
        config_id: config id used to save the model
        df_test: dataframe with the test set
        n_bootstrap: number of bootstrap iterations
        min_diff: boolean for deciding which criterion to use for selecting model from search grid
        importance: boolean to indicate to compute the feature importance or not
        list_cv: list of (train_idx, val_idx) tuples for spatial CV
        output_dir: directory to save grid search results CSV

    Returns:
        model: model fit on full training set
        bootstrap_models_info: scores and predictions for the n_bootstrap models
        results_best_model: best results of the grid search
        index_bootstrap: index for each bootstrap id model
        grid: full grid search results
    """
    print("start training model spatial cv")
    estimator = xgb.XGBRegressor(objective="reg:logistic", n_jobs=2)

    grid = GridSearchCV(
        verbose=0,
        estimator=estimator,
        scoring=["neg_mean_absolute_error", "r2", "neg_root_mean_squared_error"],
        param_grid=grid_search_dict,
        cv=list_cv,
        refit=False,
        n_jobs=2,
        return_train_score=True,
    )
    grid.fit(
        X=df_train_and_val[independent_variables],
        y=df_train_and_val[target_variable],
    )
    grid_search_full_results = pd.DataFrame(grid.cv_results_)
    csv_filename = f"{target_variable}_{last}_{config_id}_{datetime.now().strftime('%d-%m-%Y')}_{datetime.now().strftime('%H-%M-%S')}.csv"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, csv_filename)
    else:
        csv_path = csv_filename
    grid_search_full_results.to_csv(csv_path)

    results_best_model = get_tuning_best_results(grid=grid, min_diff=min_diff)
    model, _ = train_single_model(
        test=df_test,
        train=df_train_and_val,
        independent_variables=independent_variables,
        target_variable=target_variable,
        hyperparameters=results_best_model.params,
    )

    index_bootstrap = pd.DataFrame()
    pred_boots = pd.DataFrame()
    bootstrap_models_info = {}
    if n_bootstrap != 0 and n_bootstrap is not None:
        for boot_id in range(n_bootstrap):
            df_boot = df_train_and_val.sample(frac=1, replace=True, random_state=boot_id).reset_index()
            model_boot, _ = train_single_model(
                df_boot,
                df_test,
                independent_variables,
                target_variable=target_variable,
                hyperparameters=results_best_model.params,
            )
            index_bootstrap[boot_id] = df_boot["key"]

            if importance:
                imp = get_model_importance(model_boot, boot_id)
                imp.to_csv(f"importance_{target_variable}_{last}.csv")

            if df_test is not None:
                pred_boots[boot_id] = model_boot.predict(df_test[independent_variables])

        if df_test is not None:
            pred_median = pred_boots.median(axis=1)
            pred_lower = pred_boots.quantile(0.025, axis=1)
            pred_upper = pred_boots.quantile(0.975, axis=1)
            bootstrap_models_info = {
                "r2_test_boot": r2_score(y_pred=pred_median, y_true=df_test[target_variable]),
                "mae_test_boot": mean_absolute_error(y_pred=pred_median, y_true=df_test[target_variable]),
                "rmse_test_boot": root_mean_squared_error(y_pred=pred_median, y_true=df_test[target_variable]),
                "pred_median": pred_median,
                "pred_lower": pred_lower,
                "pred_upper": pred_upper,
                "pred_boots": pred_boots,
            }

    print("end training model spatial cv")
    return model, bootstrap_models_info, results_best_model, index_bootstrap, grid


# ============================================================================
# CONSTANTS
# ============================================================================

Y_NAME = "fcs"
META_COLS = ["adm1_name", "cv_ind", Y_NAME]

# Hyperparameter grid — consistent with xg-boost-temp approach
PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
}


# ============================================================================
# get_cv_splits — identical to mlp.py
# ============================================================================

def get_cv_splits(df, cv_fold):
    """
    Create custom train/validation splits for nested CV.

    Identical to mlp.py.

    Args:
        df: DataFrame with 'cv_ind' column
        cv_fold: CV fold number (0-5)

    Returns:
        train_fold: Dict of training DataFrames for each inner config
        val_fold  : Dict of validation DataFrames for each inner config
    """
    cv = {i: df[df["cv_ind"] == i] for i in range(6)}

    train_fold = {}
    val_fold = {}

    if cv_fold == 0:
        train_fold[0] = pd.concat([cv[1], cv[2], cv[3], cv[4]])
        val_fold[0] = cv[5]
        train_fold[1] = pd.concat([cv[1], cv[2], cv[3], cv[5]])
        val_fold[1] = cv[4]
        train_fold[2] = pd.concat([cv[1], cv[2], cv[4], cv[5]])
        val_fold[2] = cv[3]
        train_fold[3] = pd.concat([cv[2], cv[3], cv[4], cv[5]])
        val_fold[3] = cv[1]
        train_fold[4] = pd.concat([cv[1], cv[3], cv[4], cv[5]])
        val_fold[4] = cv[2]

    elif cv_fold == 1:
        train_fold[0] = pd.concat([cv[0], cv[2], cv[3], cv[5]])
        val_fold[0] = cv[4]
        train_fold[1] = pd.concat([cv[0], cv[3], cv[4], cv[5]])
        val_fold[1] = cv[2]
        train_fold[2] = pd.concat([cv[0], cv[2], cv[4], cv[5]])
        val_fold[2] = cv[3]
        train_fold[3] = pd.concat([cv[0], cv[2], cv[3], cv[4]])
        val_fold[3] = cv[5]
        train_fold[4] = pd.concat([cv[2], cv[3], cv[4], cv[5]])
        val_fold[4] = cv[0]

    elif cv_fold == 2:
        train_fold[0] = pd.concat([cv[0], cv[1], cv[3], cv[5]])
        val_fold[0] = cv[4]
        train_fold[1] = pd.concat([cv[0], cv[3], cv[4], cv[5]])
        val_fold[1] = cv[1]
        train_fold[2] = pd.concat([cv[0], cv[1], cv[4], cv[5]])
        val_fold[2] = cv[3]
        train_fold[3] = pd.concat([cv[0], cv[1], cv[3], cv[4]])
        val_fold[3] = cv[5]
        train_fold[4] = pd.concat([cv[1], cv[3], cv[4], cv[5]])
        val_fold[4] = cv[0]

    elif cv_fold == 3:
        train_fold[0] = pd.concat([cv[0], cv[1], cv[2], cv[5]])
        val_fold[0] = cv[4]
        train_fold[1] = pd.concat([cv[0], cv[2], cv[4], cv[5]])
        val_fold[1] = cv[1]
        train_fold[2] = pd.concat([cv[0], cv[1], cv[4], cv[5]])
        val_fold[2] = cv[2]
        train_fold[3] = pd.concat([cv[0], cv[1], cv[2], cv[4]])
        val_fold[3] = cv[5]
        train_fold[4] = pd.concat([cv[1], cv[2], cv[4], cv[5]])
        val_fold[4] = cv[0]

    elif cv_fold == 4:
        train_fold[0] = pd.concat([cv[0], cv[1], cv[3], cv[5]])
        val_fold[0] = cv[2]
        train_fold[1] = pd.concat([cv[0], cv[3], cv[2], cv[5]])
        val_fold[1] = cv[1]
        train_fold[2] = pd.concat([cv[0], cv[1], cv[2], cv[5]])
        val_fold[2] = cv[3]
        train_fold[3] = pd.concat([cv[0], cv[1], cv[3], cv[2]])
        val_fold[3] = cv[5]
        train_fold[4] = pd.concat([cv[1], cv[2], cv[3], cv[5]])
        val_fold[4] = cv[0]

    elif cv_fold == 5:
        train_fold[0] = pd.concat([cv[0], cv[1], cv[2], cv[3]])
        val_fold[0] = cv[4]
        train_fold[1] = pd.concat([cv[0], cv[1], cv[2], cv[4]])
        val_fold[1] = cv[3]
        train_fold[2] = pd.concat([cv[0], cv[1], cv[3], cv[4]])
        val_fold[2] = cv[2]
        train_fold[3] = pd.concat([cv[1], cv[2], cv[3], cv[4]])
        val_fold[3] = cv[0]
        train_fold[4] = pd.concat([cv[0], cv[1], cv[2], cv[3]])
        val_fold[4] = cv[4]

    return train_fold, val_fold


# ============================================================================
# FEATURE PREPARATION — same logic as mlp.py
# ============================================================================

def prepare_features(df, feature_cols):
    """
    Build the feature matrix from a DataFrame.

    All columns in feature_cols are kept; Datetime is converted to a Unix
    timestamp (integer seconds) so it can be used as a numeric feature,
    identical to mlp.py.

    Args:
        df         : source DataFrame
        feature_cols: list of column names to use as features

    Returns:
        X : DataFrame with numeric features
    """
    X = df[feature_cols].copy()
    X["Datetime"] = pd.to_datetime(X["Datetime"]).astype("int64") // 10 ** 9
    return X


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run XGBoost with Nested CV")
    parser.add_argument(
        "--country", type=str, required=True,
        help="Country name (e.g., chad, nigeria)"
    )
    parser.add_argument(
        "--project-path", type=str, required=True,
        help="Path to directory containing data_train_cvX.csv and data_test_cvX.csv files"
    )
    parser.add_argument(
        "--output-data", type=str, default=None,
        help="Path to save predictions CSV files and summary JSON (default: None)"
    )
    parser.add_argument(
        "--cv-folds", type=str, default="0,1,2,3,4,5",
        help="Comma-separated CV fold indices to run (default: 0,1,2,3,4,5)"
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=50,
        help="Number of bootstrap iterations (default: 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    args.project_path = os.path.normpath(os.path.join(base_dir, args.project_path))
    if args.output_data is not None:
        args.output_data = os.path.normpath(os.path.join(base_dir, args.output_data))

    cv_folds = [int(x) for x in args.cv_folds.split(",")]

    print(f"Running XGBoost Nested CV for {args.country.upper()}")
    print(f"Project path  : {args.project_path}")
    print(f"Output data   : {args.output_data if args.output_data else 'Not saving'}")
    print(f"CV Folds      : {cv_folds}")
    print(f"Bootstrap iter: {args.n_bootstrap}")

    all_results = []
    all_predictions = []

    for cv_fold in cv_folds:
        print(f"\n{'#' * 80}")
        print(f"# OUTER CV FOLD {cv_fold}")
        print(f"{'#' * 80}")

        # ------------------------------------------------------------------
        # Load data — same as mlp.py
        # ------------------------------------------------------------------
        train_path = f"{args.project_path}/data_train_cv{cv_fold}.csv"
        df = pd.read_csv(train_path)
        print(f"Loaded training data: {df.shape}")

        test_path = f"{args.project_path}/data_test_cv{cv_fold}.csv"
        test_df = pd.read_csv(test_path)
        print(f"Loaded test data    : {test_df.shape}")

        # Feature columns = all columns except metadata (same as mlp.py)
        feature_cols = [c for c in df.columns if c not in META_COLS]

        # ------------------------------------------------------------------
        # Inner CV splits from get_cv_splits — same as mlp.py
        # ------------------------------------------------------------------
        train_fold, val_fold = get_cv_splits(df, cv_fold)

        # Convert to (train_idx, val_idx) list for GridSearchCV.
        # Indices are positional in df (0..N-1 from pd.read_csv default index).
        list_cv = [
            (train_fold[j].index.values, val_fold[j].index.values)
            for j in range(5)
        ]

        # ------------------------------------------------------------------
        # Build model DataFrames
        # train_model_bootstrap_statial_cv requires:
        #   - a 'key' column (used for bootstrap index tracking)
        #   - feature columns + target column in the same DataFrame
        # ------------------------------------------------------------------
        df_train = prepare_features(df, feature_cols)
        df_train[Y_NAME] = df[Y_NAME].values
        df_train["key"] = df_train.index   # required by bootstrap function

        df_test_model = prepare_features(test_df, feature_cols)
        df_test_model[Y_NAME] = test_df[Y_NAME].values
        df_test_model["key"] = df_test_model.index

        independent_variables = [c for c in df_train.columns if c not in [Y_NAME, "key"]]

        # ------------------------------------------------------------------
        # Train XGBoost — xg-boost-temp approach
        # (GridSearchCV with inner spatial CV + bootstrap)
        # ------------------------------------------------------------------
        (
            model,
            bootstrap_models_info,
            results_best_model,
            index_bootstrap,
            grid,
        ) = train_model_bootstrap_statial_cv(
            df_train_and_val=df_train,
            df_test=df_test_model,
            independent_variables=independent_variables,
            target_variable=Y_NAME,
            last=False,
            config_id=f"{args.country}_fold{cv_fold}",
            grid_search_dict=PARAM_GRID,
            n_bootstrap=args.n_bootstrap,
            min_diff=True,
            importance=False,
            list_cv=list_cv,
            output_dir=args.output_data,
        )

        y_pred = model.predict(df_test_model[independent_variables])

        # ------------------------------------------------------------------
        # Build results DataFrame (same structure as mlp.py)
        # ------------------------------------------------------------------
        df_plot = pd.DataFrame({
            "Datetime":  pd.to_datetime(test_df["Datetime"]),
            "y_hat":     y_pred,
            "y_true":    test_df[Y_NAME].values,
            "adm1_name": test_df["adm1_name"].values,
            "cv_fold":   cv_fold,
        })

        # Add confidence intervals from bootstrap if available
        if "pred_lower" in bootstrap_models_info:
            df_plot["ci_low"] = bootstrap_models_info["pred_lower"].values
            df_plot["ci_high"] = bootstrap_models_info["pred_upper"].values

        # RMSE by region
        rmse_by_region = df_plot.groupby("adm1_name", group_keys=False).apply(
            lambda g: np.sqrt(np.mean((g["y_hat"] - g["y_true"]) ** 2))
        )
        df_plot = df_plot.merge(
            rmse_by_region.rename("rMSE"), left_on="adm1_name", right_index=True
        )

        test_rmse = float(np.sqrt(np.mean((y_pred - test_df[Y_NAME].values) ** 2)))
        print(f"\nTest RMSE = {test_rmse:.6f}")

        rmse_boot = bootstrap_models_info.get("rmse_test_boot", None)
        if rmse_boot is not None:
            print(f"Bootstrap RMSE = {rmse_boot:.6f}")

        all_predictions.append(df_plot)
        all_results.append({
            "cv_fold":    cv_fold,
            "test_rmse":  test_rmse,
            "best_params": dict(results_best_model["params"]),
        })

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'#' * 80}")
    print(f"# FINAL SUMMARY — {args.country.upper()}")
    print(f"{'#' * 80}")
    for result in all_results:
        print(
            f"CV Fold {result['cv_fold']}: Test RMSE = {result['test_rmse']:.6f} | "
            f"Best params = {result['best_params']}"
        )

    mean_rmse = float(np.mean([r["test_rmse"] for r in all_results]))
    std_rmse  = float(np.std([r["test_rmse"]  for r in all_results]))
    print(f"\nOverall Test RMSE: {mean_rmse:.6f} ± {std_rmse:.6f}")

    # -------------------------------------------------------------------------
    # Save predictions — same structure as mlp.py
    # -------------------------------------------------------------------------
    if args.output_data is not None:
        os.makedirs(args.output_data, exist_ok=True)

        # Combined predictions from all folds
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        combined_file = os.path.join(
            args.output_data, f"{args.country}_all_predictions.csv"
        )
        combined_predictions.to_csv(combined_file, index=False)
        print(f"\nCombined predictions saved to {combined_file}")

        # Per-fold files
        for df_fold in all_predictions:
            fold = int(df_fold["cv_fold"].iloc[0])
            fold_file = os.path.join(args.output_data, f"predictions_cv{fold}.csv")
            df_fold.to_csv(fold_file, index=False)
            print(f"Fold {fold} predictions saved to {fold_file}")

        # Summary JSON
        summary = {
            "country":        args.country,
            "mean_test_rmse": mean_rmse,
            "std_test_rmse":  std_rmse,
            "cv_folds": [
                {
                    "cv_fold":    int(r["cv_fold"]),
                    "test_rmse":  r["test_rmse"],
                    "best_params": r["best_params"],
                }
                for r in all_results
            ],
        }
        summary_file = os.path.join(
            args.output_data, f"{args.country}_summary.json"
        )
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_file}")

        # Overall metrics
        overall_rmse = float(np.sqrt(np.mean(
            (combined_predictions["y_hat"] - combined_predictions["y_true"]) ** 2
        )))
        print(f"\nTotal predictions : {len(combined_predictions)}")
        print(f"Number of regions : {combined_predictions['adm1_name'].nunique()}")
        print(
            f"Date range        : {combined_predictions['Datetime'].min()} "
            f"to {combined_predictions['Datetime'].max()}"
        )
        print(f"Overall RMSE      : {overall_rmse:.6f}")


if __name__ == "__main__":
    main()
