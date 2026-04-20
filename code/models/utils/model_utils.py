import os
import numpy as np
from datetime import datetime
import dateutil
import pickle

from dateutil.relativedelta import relativedelta
from typing import Dict, Optional, Union, List


import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

BASE_DIR = os.getcwd()
GSR_DIR = os.path.join(BASE_DIR, "grid_search_results")

# ===============================================
# MODEL FIT UTILITIES FUNCTIONS
# ===============================================

def logit(x):
    return np.log(x / (1 - x))

def logit_inverse(x):
    return np.exp(x) / (1 + np.exp(x))


def train_single_model(
    train: pd.DataFrame,
    test: Optional[pd.DataFrame],
    independent_variables: List[str],
    target_variable: str,
    hyperparameters: Dict[str, int],
):
    """
    Trains a single xgboost modest performing cross validation

    Args:
        train: train set of the data
        test: test set of the data
        independent_variables: list of explanatory variables
        target_variable: name of the dependent variable
        hyperparameters: hyperparameters to be used in the fit

    Returns:
        model: fit model
        results: overview of the fit model
    """
    # Fit model with chosen hyperparams
    model = xgb.XGBRegressor(
        verbosity=0, **hyperparameters, n_jobs=2, objective="reg:logistic"
    )
    model.fit(train[independent_variables], train[target_variable])

    y_predict_train = model.predict(train[independent_variables])
    r2_train = r2_score(y_pred=y_predict_train, y_true=train[target_variable])
    mae_train = mean_absolute_error(
        y_pred=y_predict_train, y_true=train[target_variable]
    )

    if test is not None:
        y_predict_test = model.predict(test[independent_variables])
        r2_test = r2_score(y_pred=y_predict_test, y_true=test[target_variable])
        mae_test = mean_absolute_error(
            y_pred=y_predict_test, y_true=test[target_variable]
        )

        results = {
            "r2_test": r2_test,
            "mae_test": mae_test,
            "r2_train": r2_train,
            "mae_train": mae_train,
        }
    else:
        results = {
            "r2_train": r2_train,
            "mae_train": mae_train,
        }

    return model, results


def train_model_bootstrap(
    df_train_and_val: pd.DataFrame,
    independent_variables: Union[list, set],
    target_variable: str,
    last: bool,
    grid_search_dict: Dict,
    config_id: Union[int, str],
    months_aggregated_val: int,
    number_of_splits_validation: int,
    df_test: Optional[Union[pd.DataFrame, list]] = None,
    n_bootstrap: int = 100,
    min_diff: bool = True,
    save_model: bool = False,
    time_column_name: str = "date",
    importance: bool = False,
):

    """
    Performs a grid search and trains n_bootstrap models by re-sampling the provided dataset
    Args:
        df_train_and_val: dataframe with training and validation set
        independent_variables: list of explanatory variables
        target_variable: name of the dependent variable
        last: boolean variable which indicates if the previous recorded score
         of the dependent variable is used as explanatory variable
        grid_search_dict: hyperparameters over the grid search is performed
        config_id: config id used to save the model
        months_aggregated_val: number of months aggregated in each split of the walk forward val
        number_of_splits_validation: number of splits in the walk forward validation
        df_test: dataframe with the test set
        n_bootstrap: number of bootstrap iterations
        min_diff: boolean for deciding which criterion to use for selecting model from search grid
        save_model: boolean to indicate whether to save the model or not. It is saved as pickle
        time_column_name: name of the column on which the walk forward splits are performed
        importance: boolean to indicate to compute the feature importance or not

    Returns:
        model: model fit on full training set
        bootstrap_models_inf0: scores for the n_bootstrap models
        results_best_model: best results of the grid search
        index_bootstrap: index for each bootstrap id model. These are needed for the shap values calculation
        grid: full grid search results
    """
    # Define model
    estimator = xgb.XGBRegressor(objective="reg:logistic", n_jobs=2)
    # Get hyperparameters
    list_cv = get_temporal_validation_idx(
        df_train_and_val=df_train_and_val,
        time_column_name=time_column_name,
        months_aggregated=months_aggregated_val,
        number_of_splits=number_of_splits_validation,
    )

    grid = GridSearchCV(
        verbose=0,
        estimator=estimator,
        scoring=["neg_mean_absolute_error", "r2"],
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
    grid_search_full_results.to_csv(
        f"{target_variable}_{last}_{config_id}_{datetime.now().strftime('%d-%m-%Y')}_{datetime.now().strftime('%H-%M-%S')}.csv",
    )
    results_best_model = get_tuning_best_results(grid=grid, min_diff=min_diff)
    # fit single model on train + val
    model, res = train_single_model(
        test=df_test,
        train=df_train_and_val,
        independent_variables=independent_variables,
        target_variable=target_variable,
        hyperparameters=results_best_model.params,
    )
    if save_model:
        save_model_pickle(
            model=model,
            last=last,
            label=target_variable,
            id=config_id,
            bootstrap_id=-1,
        )
    # bootstrap
    index_bootstrap = pd.DataFrame()
    pred_boots = pd.DataFrame()
    bootstrap_models_info = {}
    if n_bootstrap != 0 and n_bootstrap is not None:
        for boot_id in range(n_bootstrap):
            # resample with replacement
            df_boot = df_train_and_val.sample(
                frac=1, replace=True, random_state=boot_id
            ).reset_index()
            # train the model
            model_boot, res = train_single_model(
                df_boot,
                df_test,
                independent_variables,
                target_variable=target_variable,
                hyperparameters=results_best_model.params,
            )
            index_bootstrap[boot_id] = df_boot["key"]
            if save_model:
                save_model_pickle(
                    model=model,
                    last=last,
                    label=target_variable,
                    id=config_id,
                    bootstrap_id=boot_id,
                )
            # Get importance
            if importance:
                imp = get_model_importance(model_boot, boot_id)
                imp.to_csv(f"importance_{target_variable}_{last}.csv")

            if df_test is not None:
                pred_boot = model_boot.predict(df_test[independent_variables])
                pred_boots[boot_id] = pred_boot

        if df_test is not None:
            pred_median = pred_boots.median(axis=1)
            r2_test_boot = r2_score(
                y_pred=pred_median, y_true=df_test[f"{target_variable}"]
            )
            mae_test_boot = mean_absolute_error(
                y_pred=pred_median, y_true=df_test[f"{target_variable}"]
            )

            bootstrap_models_info = {
                "r2_test_boot": r2_test_boot,
                "mae_test_boot": mae_test_boot,
            }

    return model, bootstrap_models_info, results_best_model, index_bootstrap, grid


def save_model_pickle(model, label: str, last: bool, id: str, bootstrap_id: int):
    """
    Pulls the specified model

    Args:
        model: model to be saved
        label: output variable. `fcs` or `rcsi`.
        last: boolean variable which indicates if the previous score of the dependent variable
         is used as explanatory variable
        id: config_id
        bootstrap_id: the bootstrapped model id. If `-1` is the model trained on the original training set

    """
    path = "/XGBoost_{}_{}_{}/XGBoost_{}_{}_{}_{}.dat".format(
        label, id, last, label, id, last, bootstrap_id
    )
    with open(path, "wb") as handle:

        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pull_model(label: str, last: bool, id: str, bootstrap_id: int):
    """
    Pulls the specified model

    Args:
        label: output variable. `fcs` or `rcsi`.
        last: boolean variable which indicates if the previous recorded score
         of the dependent variable is used as explanatory variable
        id: config_id
        bootstrap_id: the bootstrapped model id. If `-1` is the model trained on the original training set

    Returns:
        model: model pulled. It has a predict method

    """
    path = "model/XGBoost_{}_{}_{}/XGBoost_{}_{}_{}_{}.dat".format(
        label, id, last, label, id, last, bootstrap_id
    )
    with (open(path, "rb")) as model_file:
        model = pickle.load(model_file)

    return model


def get_temporal_validation_idx(
    df_train_and_val: pd.DataFrame,
    time_column_name: str,
    months_aggregated: int = 1,
    number_of_splits: int = 5,
) -> List[tuple[np.ndarray, np.ndarray]]:
    """

    Args:
        df_train_and_val: dataframe containing train and validation set
        time_column_name: name of the column on which the temporal splits are created. It should contain dates
        months_aggregated: number of months aggregated in each split
        number_of_splits: number of splits to be created

    Returns:
        list_cv: list with indexes for each validation split
    """
    df_train_and_val.loc[:, time_column_name] = pd.to_datetime(
        df_train_and_val.loc[:, time_column_name]
    )
    last_date = df_train_and_val[time_column_name].max()
    first_day_month_last_date = last_date.replace(day=1)
    list_cv = []
    for ii in range(number_of_splits):
        first_day_validation = first_day_month_last_date - relativedelta(
            months=(number_of_splits - ii) * months_aggregated - 1
        )
        last_day_validation = first_day_validation + relativedelta(
            months=months_aggregated
        )
        df_validation = df_train_and_val.loc[
            (df_train_and_val[time_column_name] >= first_day_validation)
            & (df_train_and_val[time_column_name] < last_day_validation)
        ]
        df_train = df_train_and_val.loc[
            df_train_and_val[time_column_name] < first_day_validation
        ]
        list_cv.append(
            (
                np.array(df_train.index.values.astype(int)),
                np.array(df_validation.index.values.astype(int)),
            )
        )
        # yield df_train.index.values.astype(int), df_validation.index.values.astype(int)
    return list_cv


def get_tuning_best_results(
    grid: GridSearchCV, min_diff: bool, lambda_parameter: Optional[float] = 1
) -> pd.DataFrame:
    """

    Args:
        grid:
        min_diff: boolean to indicate which criterion should be used to get the best hyperparams.
         If true the hyperparams with the minimum diff between train and validation set are chosen
        lambda_parameter: scaling factor if min_diff is True.

    Returns:
        results_best_model: combination of best hyperparams according to the criterion selected
    """
    grid_search_full_results = pd.DataFrame(grid.cv_results_)
    grid_search_full_results["r2_diff"] = (
        grid_search_full_results["mean_train_r2"]
        - grid_search_full_results["mean_test_r2"]
    )

    if min_diff:
        grid_search_full_results["elaborated_function"] = lambda_parameter * (
            grid_search_full_results["mean_train_r2"]
            - grid_search_full_results["mean_test_r2"]
        ) + (1 - lambda_parameter) * (1.0 - grid_search_full_results["mean_test_r2"])

        r2_min_diff_vec = []
        for index, row in grid_search_full_results.iterrows():
            if row["r2_diff"] >= 0:
                r2_min_diff_vec.append(row["elaborated_function"])
        if len(r2_min_diff_vec) > 0:
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
        results_best_model_split = grid_search_full_results.loc[
            grid_search_full_results["rank_test_r2"] == 1
        ].iloc[0]
    return results_best_model_split


def get_model_importance(model, boot_id: int):
    """

    Args:
        model:
        boot_id:

    Returns:

    """
    importance = model.get_booster().get_score(importance_type="total_gain")
    imp = pd.DataFrame.from_dict(importance, orient="index")
    imp.reset_index(level=0, inplace=True)
    imp.columns = ["feature", "value"]
    imp["value"] = imp["value"] / imp["value"].sum() * 100
    imp = imp.sort_values(by=["value"], ascending=False, axis=0).reset_index(drop=True)
    imp["iteration"] = boot_id
    return imp


def get_temporal_splits(
    df: pd.DataFrame,
    time_column_name: str,
    number_of_splits: int = 5,
    months_aggregated: int = 1,
):
    """

    Args:
        df:
        time_column_name:
        number_of_splits:
        months_aggregated:

    Returns:

    """
    """If n splits are desired the a dictionary with the keys of the last n months is returned
    The keys of the dictionary are numbers ranging from 1 to 5, the key 1 refers to the oldest of the n months"""

    df.loc[:, time_column_name] = pd.to_datetime(df.loc[:, time_column_name])
    last_date = df[time_column_name].max()
    first_day_month_last_date = last_date.replace(day=1)
    split_dates = {}
    keys_test = {}
    for ii in np.arange(number_of_splits):
        split_dates[
            ii
        ] = first_day_month_last_date - dateutil.relativedelta.relativedelta(
            months=number_of_splits * months_aggregated - months_aggregated * ii - 1
        )
    test_points = df[df[time_column_name] >= split_dates[0]]
    for ii in np.arange(number_of_splits):
        if ii != number_of_splits - 1:
            keys_test[ii] = test_points.loc[
                (split_dates[ii] <= test_points[time_column_name])
                & (test_points[time_column_name] < split_dates[ii + 1]),
                "key",
            ]
        else:
            keys_test[ii] = test_points.loc[
                test_points[time_column_name] >= split_dates[ii], "key"
            ]

    return keys_test, split_dates


def sample_training_data_cm(data: pd.DataFrame, n_days_per_month: int = 1):
    """

    Args:
        data: historical and cm data
        n_days_per_month: number of days per month to be sampled per adm1

    Returns:
        data: sampled data

    """
    data["date"] = pd.to_datetime(data["date"])
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month

    data_historical = data.loc[data["is_historical"] is True, :]
    data_cm = data.loc[data["is_historical"] is not True, :]

    data_cm_fcs = data_cm[~data_cm.fcs.isna()]
    data_cm_rcsi = data_cm[~data_cm.rcsi.isna()]
    if len(data_cm_fcs) > 0:
        data_sampled_fcs = data_cm_fcs.groupby(["adm1_code", "year", "month"]).sample(
            n=n_days_per_month, replace=True, random_state=4
        )
    else:
        data_sampled_fcs = data_cm_fcs
    if len(data_cm_rcsi) > 0:
        data_sampled_rcsi = data_cm_rcsi.groupby(["adm1_code", "year", "month"]).sample(
            n=n_days_per_month, replace=True, random_state=4
        )
    else:
        data_sampled_rcsi = data_cm_rcsi
    data_sampled_fcs.drop_duplicates(
        subset=["date", "adm1_code"], keep="first", inplace=True
    )
    data_sampled_rcsi.drop_duplicates(
        subset=["date", "adm1_code"], keep="first", inplace=True
    )
    data = pd.concat([data_sampled_rcsi, data_sampled_fcs, data_historical])
    return data
