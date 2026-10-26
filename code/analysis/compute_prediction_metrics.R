# ============================================================================
# Compute prediction metrics (RMSE, MAE, 95% coverage, 95% interval length,
# CRPS, interval score) for all models, for Nigeria and Chad, and save as
# LaTeX tables matching output/tables/{tcd,nga}_prediction_metrics.tex.
#
# CRPS and interval score definitions follow
# code/analysis/proper_scoring_rules_definitions.tex:
#   - CRPS: empirical estimator (scoringRules::crps_sample) from model draws
#   - Interval score: IS_alpha at alpha = 0.05 (95% interval)
#
# Per-model draws (added alongside the existing point/interval predictions):
#   - GP:             ppc_samples_cv*.csv (joined to ppc_summary_cv*.csv)
#   - Bayesian Ridge:  *_mc_samples_fold*.csv (joined to *_predictions_fold*.csv)
#   - MLP:             mc_samples_cv*.csv (joined to predictions_cv*.csv)
#   - XGBoost:         bootstrap_preds_cv*.csv (joined to predictions_cv*.csv)
# ============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(scoringRules)
  library(here)
})

topdir <- here::here()

# ============================================================================
# Interval score (Gneiting & Raftery, 2007), alpha = 0.05 (95% interval)
# ============================================================================
interval_score <- function(y, lower, upper, alpha = 0.05) {
  (upper - lower) +
    (2 / alpha) * (lower - y) * (y < lower) +
    (2 / alpha) * (y - upper) * (y > upper)
}

crps_from_draws <- function(y_true, draws) {
  crps_sample(y = y_true, dat = draws)
}

# ============================================================================
# Per-model metric extraction: each function returns one row per observation
# with y_true, y_hat, ci_low, ci_high, crps
# ============================================================================

metrics_gp <- function(run_dir) {
  diag_dir <- file.path(run_dir, "prior_posterior_diagnostics")
  sample_files <- list.files(diag_dir, pattern = "^ppc_samples_cv\\d+\\.csv$", full.names = TRUE)

  map_dfr(sample_files, function(f) {
    cv_fold <- as.integer(gsub("\\D", "", basename(f)))
    if (cv_fold == 0) return(NULL) # fold 0 is never a held-out test fold

    samples <- read_csv(f, show_col_types = FALSE)
    summary <- read_csv(file.path(diag_dir, paste0("ppc_summary_cv", cv_fold, ".csv")), show_col_types = FALSE)

    sample_cols <- grep("^sample_", names(samples), value = TRUE)
    draws <- as.matrix(samples[, sample_cols])

    tibble(
      y_true = summary$y_true,
      y_hat = summary$y_pred_mean,
      ci_low = summary$y_pred_ci_2.5,
      ci_high = summary$y_pred_ci_97.5,
      crps = crps_from_draws(summary$y_true, draws)
    )
  })
}

metrics_bayesian_ridge <- function(data_dir, prefix) {
  sample_files <- list.files(data_dir, pattern = paste0("^", prefix, "_mc_samples_fold\\d+\\.csv$"), full.names = TRUE)

  map_dfr(sample_files, function(f) {
    fold <- as.integer(gsub("\\D", "", basename(f)))
    if (fold == 0) return(NULL) # fold 0 is never a held-out test fold

    samples <- read_csv(f, show_col_types = FALSE)
    preds <- read_csv(file.path(data_dir, paste0(prefix, "_predictions_fold", fold, ".csv")), show_col_types = FALSE)

    sample_cols <- grep("^sample_", names(samples), value = TRUE)
    draws <- as.matrix(samples[, sample_cols])

    tibble(
      y_true = preds$y_true,
      y_hat = preds$y_hat,
      ci_low = preds$ci_low,
      ci_high = preds$ci_high,
      crps = crps_from_draws(samples$y_true, draws)
    )
  })
}

metrics_mlp <- function(data_dir) {
  sample_files <- list.files(data_dir, pattern = "^mc_samples_cv\\d+\\.csv$", full.names = TRUE)

  map_dfr(sample_files, function(f) {
    fold <- as.integer(gsub("\\D", "", basename(f)))
    if (fold == 0) return(NULL) # fold 0 is never a held-out test fold

    samples <- read_csv(f, show_col_types = FALSE)
    preds <- read_csv(file.path(data_dir, paste0("predictions_cv", fold, ".csv")), show_col_types = FALSE)

    sample_cols <- grep("^sample_", names(samples), value = TRUE)
    draws <- as.matrix(samples[, sample_cols])

    tibble(
      y_true = preds$y_true,
      y_hat = preds$y_hat,
      ci_low = preds$ci_low,
      ci_high = preds$ci_high,
      crps = crps_from_draws(samples$y_true, draws)
    )
  })
}

metrics_xgboost <- function(data_dir) {
  sample_files <- list.files(data_dir, pattern = "^bootstrap_preds_cv\\d+\\.csv$", full.names = TRUE)

  map_dfr(sample_files, function(f) {
    fold <- as.integer(gsub("\\D", "", basename(f)))
    if (fold == 0) return(NULL) # fold 0 is never a held-out test fold

    samples <- read_csv(f, show_col_types = FALSE)
    preds <- read_csv(file.path(data_dir, paste0("predictions_cv", fold, ".csv")), show_col_types = FALSE)

    sample_cols <- grep("^sample_", names(samples), value = TRUE)
    draws <- as.matrix(samples[, sample_cols])

    tibble(
      y_true = preds$y_true,
      y_hat = preds$y_hat,
      ci_low = preds$ci_low,
      ci_high = preds$ci_high,
      crps = crps_from_draws(samples$y_true, draws)
    )
  })
}

# ============================================================================
# Summarise one model's per-observation metrics into a single table row
# ============================================================================
summarise_model <- function(df, model_name) {
  df <- df |> filter(!is.na(y_true), !is.na(y_hat))
  tibble(
    Model = model_name,
    RMSE = sqrt(mean((df$y_hat - df$y_true)^2)),
    MAE = mean(abs(df$y_hat - df$y_true)),
    `Coverage 95` = mean(df$y_true >= df$ci_low & df$y_true <= df$ci_high),
    `Length IC 95` = mean(df$ci_high - df$ci_low),
    CRPS = mean(df$crps),
    `Interval Score 95` = mean(interval_score(df$y_true, df$ci_low, df$ci_high))
  )
}

# ============================================================================
# Write a LaTeX table matching the existing format
# ============================================================================
write_metrics_table <- function(df, caption, label, out_path) {
  df <- df |> arrange(Model)

  fmt <- function(x) sprintf("%.3f", x)
  body <- df |>
    mutate(across(-Model, fmt)) |>
    unite("row", everything(), sep = " & ") |>
    pull(row) |>
    paste(collapse = " \\\\\n")

  header <- paste(names(df), collapse = " & ")
  col_spec <- paste0("l", strrep("r", ncol(df) - 1))

  tex <- paste0(
    "\\begin{table}\n",
    "\\caption{", caption, "}\n",
    "\\label{", label, "}\n",
    "\\begin{tabular}{", col_spec, "}\n",
    "\\toprule\n",
    header, " \\\\\n",
    "\\midrule\n",
    body, " \\\\\n",
    "\\bottomrule\n",
    "\\end{tabular}\n",
    "\\end{table}\n"
  )

  writeLines(tex, out_path)
  cat("Saved", out_path, "\n")
}

# ============================================================================
# Run for one country
# ============================================================================
run_country <- function(country_label, gp_with_features_dir, gp_no_features_dir,
                         br_dir, br_prefix, mlp_dir, xgb_dir, caption, label, out_path) {
  cat("\n==", country_label, "==\n")

  results <- bind_rows(
    summarise_model(metrics_gp(gp_with_features_dir), "GP With Features"),
    summarise_model(metrics_gp(gp_no_features_dir), "GP No Features"),
    summarise_model(metrics_bayesian_ridge(br_dir, br_prefix), "Bayesian Ridge"),
    summarise_model(metrics_mlp(mlp_dir), "MLP"),
    summarise_model(metrics_xgboost(xgb_dir), "XG-Boost")
  )

  print(results |> mutate(across(where(is.numeric), ~ round(.x, 3))), n = Inf)
  write_metrics_table(results, caption, label, out_path)

  results
}

# ============================================================================
# CONFIGURATION
# ============================================================================

tcd_metrics <- run_country(
  country_label = "Chad",
  gp_with_features_dir = file.path(topdir, "output", "gaussian_process", "TCD_gp_with_features_ridge_centered_20261026_132935"),
  gp_no_features_dir = file.path(topdir, "output", "gaussian_process", "TCD_gp_no_prior_20261026_174120"),
  br_dir = file.path(topdir, "output", "data", "tcd_bayesian_ridge"),
  br_prefix = "tcd",
  mlp_dir = file.path(topdir, "output", "data", "tcd_mlp"),
  xgb_dir = file.path(topdir, "output", "data", "tcd_xg_boost"),
  caption = "Metrics for Chad models",
  label = "tcd-tab-models",
  out_path = file.path(topdir, "output", "tables", "tcd_prediction_metrics.tex")
)

nga_metrics <- run_country(
  country_label = "Nigeria",
  gp_with_features_dir = file.path(topdir, "output", "gaussian_process", "NGA_gp_with_features_normal_centered_20260217_155435"),
  gp_no_features_dir = file.path(topdir, "output", "gaussian_process", "NGA_gp_no_features_20251020_161159"),
  br_dir = file.path(topdir, "output", "data", "nga_bayesian_ridge"),
  br_prefix = "nga",
  mlp_dir = file.path(topdir, "output", "data", "nga_mlp"),
  xgb_dir = file.path(topdir, "output", "data", "nga_xg_boost"),
  caption = "Metrics for Nigeria models",
  label = "nga-tab-models",
  out_path = file.path(topdir, "output", "tables", "nga_prediction_metrics.tex")
)
