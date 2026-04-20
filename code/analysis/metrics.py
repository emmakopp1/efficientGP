import pandas as pd
from pathlib import Path
import os
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

ROOT = Path(__file__).resolve().parents[2]

# TCD
path_tcd = [
    ROOT / "output/gaussian_process/TCD_gp_with_features_ridge_centered_20251216_114735",
    ROOT / "output/data/tcd_bayesian_ridge",
    ROOT / "output/data/tcd_mlp",
    ROOT / "output/gaussian_process/TCD_gp_no_prior_20251210_112502",
    ROOT / "output/data/tcd_xg_boost",
]

# Nigeria
path_nga = [
    ROOT / "output/gaussian_process/NGA_gp_with_features_normal_centered_20260217_155435",
    ROOT / "output/data/nga_bayesian_ridge",
    ROOT / "output/data/nga_mlp",
    ROOT / "output/data/nga_xg_boost",
    ROOT / "output/gaussian_process/NGA_gp_no_features_20251020_161159",
]


def load_model_metrics_stat(model_folder: Path):
    model = str(model_folder).split("/")[-1].split("_")
    gp_index = model.index("gp")
    model_name = " ".join(model[gp_index : (gp_index + 3)]).title()

    ppc_summary_path = model_folder / "prior_posterior_diagnostics" / "ppc_summary_all_folds.csv"

    if ppc_summary_path.exists():
        metrics = (
            pd.read_csv(ppc_summary_path)
            .query("cv_fold != 0")
            .groupby(["cv_fold", "adm1_name"])[
                ["y_true", "y_pred_mean", "y_pred_ci_2.5", "y_pred_ci_97.5", "len_ic_95"]
            ]
            .apply(
                lambda x: pd.Series({
                    "RMSE": root_mean_squared_error(x["y_true"].values, x["y_pred_mean"].values),
                    "MAE": mean_absolute_error(x["y_true"].values, x["y_pred_mean"].values),
                    "Coverage 95": (
                        (x["y_true"].values >= x["y_pred_ci_2.5"].values)
                        & (x["y_true"].values <= x["y_pred_ci_97.5"].values)
                    ).mean(),
                    "Length IC 95": x["len_ic_95"].mean(),
                })
            )
            .reset_index()
            .groupby("adm1_name")[["RMSE", "MAE", "Coverage 95", "Length IC 95"]]
            .mean()
            .reset_index()
            .assign(Model=model_name)
        )
        return metrics


def load_model_metrics_mlp(model_folder: Path):
    all_metrics = []

    for i in range(1, 6):
        df = pd.read_csv(os.path.join(model_folder, f"predictions_cv{i}.csv"))
        df = df[["y_hat", "y_true", "ci_low", "ci_high", "adm1_name"]]

        metrics = (
            df.groupby("adm1_name")[["y_hat", "y_true", "ci_low", "ci_high", "adm1_name"]]
            .apply(
                lambda x: pd.Series({
                    "RMSE": root_mean_squared_error(x["y_true"], x["y_hat"]),
                    "MAE": mean_absolute_error(x["y_true"], x["y_hat"]),
                    "Coverage 95": ((x["y_true"] >= x["ci_low"]) & (x["y_true"] <= x["ci_high"])).mean(),
                    "Length IC 95": (x["ci_high"] - x["ci_low"]).mean(),
                })
            )
        )
        all_metrics.append(metrics)

    return (
        pd.concat(all_metrics)
        .mean()
        .pipe(lambda s: pd.DataFrame([s]))
        .assign(Model="MLP")
    )


def load_model_metrics_br(model_folder: Path):
    all_metrics = []
    country = str(model_folder).split("/")[-1].split("_")[0]

    for i in range(1, 6):
        df = pd.read_csv(os.path.join(model_folder, f"{country}_predictions_fold{i}.csv"))

        metrics = (
            df.groupby("adm1_name")[["y_hat", "y_true", "ci_low", "ci_high", "adm1_name"]]
            .apply(
                lambda x: pd.Series({
                    "RMSE": root_mean_squared_error(x["y_true"], x["y_hat"]),
                    "MAE": mean_absolute_error(x["y_true"], x["y_hat"]),
                    "Coverage 95": ((x["y_true"] >= x["ci_low"]) & (x["y_true"] <= x["ci_high"])).mean(),
                    "Length IC 95": (x["ci_high"] - x["ci_low"]).mean(),
                })
            )
        )
        all_metrics.append(metrics)

    return (
        pd.concat(all_metrics)
        .mean()
        .pipe(lambda s: pd.DataFrame([s]))
        .assign(Model="Bayesian Ridge")
    )


def load_model_metrics_xgboost(model_folder: Path):
    all_metrics = []

    for i in range(1, 6):
        df = pd.read_csv(os.path.join(model_folder, "predictions_cv{}.csv".format(i)))

        metrics = (
            df.groupby("adm1_name")[["y_hat", "y_true", "ci_low", "ci_high"]]
            .apply(
                lambda x: pd.Series({
                    "RMSE": root_mean_squared_error(x["y_true"], x["y_hat"]),
                    "MAE": mean_absolute_error(x["y_true"], x["y_hat"]),
                    "Coverage 95": ((x["y_true"] >= x["ci_low"]) & (x["y_true"] <= x["ci_high"])).mean(),
                    "Length IC 95": (x["ci_high"] - x["ci_low"]).mean(),
                })
            )
        )
        all_metrics.append(metrics)

    return pd.concat(all_metrics).assign(Model="XG-Boost").reset_index()


TABLES_DIR = ROOT / "output/tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Nigeria
df_gp_cov = load_model_metrics_stat(path_nga[0])
df_mlp = load_model_metrics_mlp(path_nga[2])
df_br = load_model_metrics_br(path_nga[1])
df_xg_boost = load_model_metrics_xgboost(path_nga[3])
df_gp_no_cov = load_model_metrics_stat(path_nga[4])

df_nga = (
    pd.concat([df_gp_cov, df_mlp, df_br, df_xg_boost, df_gp_no_cov], ignore_index=True)
    .groupby("Model")[["RMSE", "MAE", "Coverage 95", "Length IC 95"]]
    .mean()
    .reset_index()
)
print("Nigeria:")
print(df_nga)

df_nga.to_latex(
    TABLES_DIR / "nga_prediction_metrics.tex",
    index=False,
    caption="Metrics for Nigeria models",
    label="nga-tab-models",
    float_format="%.3f",
)

# Chad
df_gp_cov = load_model_metrics_stat(path_tcd[0])
df_gp_cov = df_gp_cov[["Model", "RMSE", "MAE", "Coverage 95", "Length IC 95"]]
df_mlp = load_model_metrics_mlp(path_tcd[2])
df_br = load_model_metrics_br(path_tcd[1])
df_gp_no_cov = load_model_metrics_stat(path_tcd[3])
df_xg_boost = load_model_metrics_xgboost(path_tcd[4])

df_tcd = (
    pd.concat([df_gp_cov, df_mlp, df_br, df_gp_no_cov, df_xg_boost], ignore_index=True)
    .groupby("Model")[["RMSE", "MAE", "Coverage 95", "Length IC 95"]]
    .mean()
    .reset_index()
    .replace("Gp No Prior", "GP no features")
)
print("\nChad:")
print(df_tcd)

df_tcd.to_latex(
    TABLES_DIR / "tcd_prediction_metrics.tex",
    index=False,
    caption="Metrics for Chad models",
    label="tcd-tab-models",
    float_format="%.3f",
)
