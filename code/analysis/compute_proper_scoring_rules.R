# ============================================================================
# Compute CRPS and interval score for all 4 models (GP, Bayesian Ridge, MLP,
# XGBoost), for Nigeria and Chad.
# ============================================================================
# Definitions follow code/analysis/proper_scoring_rules_definitions.tex:
#   - CRPS: empirical estimator (scoringRules::crps_sample) from model draws
#   - Interval score: IS_alpha at alpha = 0.05 (95% interval), from the
#     existing ci_low / ci_high columns already used for coverage reporting
#
# Expects, per model / country / fold, a "samples" file with columns
# Datetime, adm1_name, y_true, cv_fold, sample_1 ... sample_M:
#   - GP:             output/gaussian_process/<run>/prior_posterior_diagnostics/ppc_samples_cv*.csv
#                      (joined to ppc_summary_cv*.csv for adm1_name/Datetime)
#   - Bayesian Ridge:  output/data/<country>_bayesian_ridge/<iso>_mc_samples_fold*.csv
#   - MLP:             output/data/<country>_mlp/mc_samples_cv*.csv
#   - XGBoost:         output/data/<country>_xg_boost/bootstrap_preds_cv*.csv
#
# Interval score additionally needs ci_low/ci_high, read from each model's
# existing predictions_cv*.csv / ppc_summary_cv*.csv.
# ============================================================================

library(tidyverse)
library(scoringRules)
library(here)

# ============================================================================
# Interval score (Gneiting & Raftery, 2007), alpha = 0.05 (95% interval)
# ============================================================================
interval_score <- function(y, lower, upper, alpha = 0.05) {
  (upper - lower) +
    (2 / alpha) * (lower - y) * (y < lower) +
    (2 / alpha) * (y - upper) * (y > upper)
}

# ============================================================================
# Helper: CRPS from a long samples data frame
# (columns: ..., y_true, sample_1..sample_M)
# ============================================================================
crps_from_samples <- function(df) {
  sample_cols <- grep("^sample_", names(df), value = TRUE)
  draws <- as.matrix(df[, sample_cols])
  crps_sample(y = df$y_true, dat = draws)
}

# ============================================================================
# GP: read ppc_samples_cv*.csv (already FCS-scale) + ppc_summary_cv*.csv
# (for adm1_name / Datetime / ci_low / ci_high, matched by observation_id)
# ============================================================================
score_gp <- function(run_dir) {
  diag_dir <- file.path(run_dir, "prior_posterior_diagnostics")
  sample_files <- list.files(diag_dir, pattern = "^ppc_samples_cv\\d+\\.csv$", full.names = TRUE)

  map_dfr(sample_files, function(f) {
    cv_fold <- as.integer(gsub("\\D", "", basename(f)))
    samples <- read_csv(f, show_col_types = FALSE)
    summary <- read_csv(file.path(diag_dir, paste0("ppc_summary_cv", cv_fold, ".csv")), show_col_types = FALSE)

    sample_cols <- grep("^sample_", names(samples), value = TRUE)
    crps <- crps_sample(y = samples$y_true, dat = as.matrix(samples[, sample_cols]))

    tibble(
      model = "Gaussian Process",
      cv_fold = cv_fold,
      adm1_name = summary$adm1_name,
      crps = crps,
      interval_score = interval_score(summary$y_true, summary$y_pred_ci_2.5, summary$y_pred_ci_97.5)
    )
  })
}

# ============================================================================
# Bayesian Ridge: mc_samples_fold*.csv (logit-space Normal, sampled and
# back-transformed at write time) + predictions_fold*.csv (for ci_low/ci_high)
# ============================================================================
score_bayesian_ridge <- function(data_dir, prefix) {
  sample_files <- list.files(data_dir, pattern = paste0("^", prefix, "_mc_samples_fold\\d+\\.csv$"), full.names = TRUE)

  map_dfr(sample_files, function(f) {
    fold <- as.integer(gsub("\\D", "", basename(f)))
    samples <- read_csv(f, show_col_types = FALSE)
    preds <- read_csv(file.path(data_dir, paste0(prefix, "_predictions_fold", fold, ".csv")), show_col_types = FALSE)

    sample_cols <- grep("^sample_", names(samples), value = TRUE)
    crps <- crps_sample(y = samples$y_true, dat = as.matrix(samples[, sample_cols]))

    tibble(
      model = "Bayesian Ridge",
      cv_fold = fold,
      adm1_name = samples$adm1_name,
      crps = crps,
      interval_score = interval_score(preds$y_true, preds$ci_low, preds$ci_high)
    )
  })
}

# ============================================================================
# MLP: mc_samples_cv*.csv + predictions_cv*.csv (for ci_low/ci_high)
# ============================================================================
score_mlp <- function(data_dir) {
  sample_files <- list.files(data_dir, pattern = "^mc_samples_cv\\d+\\.csv$", full.names = TRUE)

  map_dfr(sample_files, function(f) {
    fold <- as.integer(gsub("\\D", "", basename(f)))
    samples <- read_csv(f, show_col_types = FALSE)
    preds <- read_csv(file.path(data_dir, paste0("predictions_cv", fold, ".csv")), show_col_types = FALSE)

    sample_cols <- grep("^sample_", names(samples), value = TRUE)
    crps <- crps_sample(y = samples$y_true, dat = as.matrix(samples[, sample_cols]))

    tibble(
      model = "MLP",
      cv_fold = fold,
      adm1_name = samples$adm1_name,
      crps = crps,
      interval_score = interval_score(preds$y_true, preds$ci_low, preds$ci_high)
    )
  })
}

# ============================================================================
# XGBoost: bootstrap_preds_cv*.csv + predictions_cv*.csv (for ci_low/ci_high)
# ============================================================================
score_xgboost <- function(data_dir) {
  sample_files <- list.files(data_dir, pattern = "^bootstrap_preds_cv\\d+\\.csv$", full.names = TRUE)

  map_dfr(sample_files, function(f) {
    fold <- as.integer(gsub("\\D", "", basename(f)))
    samples <- read_csv(f, show_col_types = FALSE)
    preds <- read_csv(file.path(data_dir, paste0("predictions_cv", fold, ".csv")), show_col_types = FALSE)

    sample_cols <- grep("^sample_", names(samples), value = TRUE)
    crps <- crps_sample(y = samples$y_true, dat = as.matrix(samples[, sample_cols]))

    tibble(
      model = "XGBoost",
      cv_fold = fold,
      adm1_name = samples$adm1_name,
      crps = crps,
      interval_score = interval_score(preds$y_true, preds$ci_low, preds$ci_high)
    )
  })
}

# ============================================================================
# Run for one country and save a summary table
# ============================================================================
score_country <- function(country_label, gp_run_dir, br_dir, br_prefix, mlp_dir, xgb_dir, out_file) {
  results <- bind_rows(
    score_gp(gp_run_dir),
    score_bayesian_ridge(br_dir, br_prefix),
    score_mlp(mlp_dir),
    score_xgboost(xgb_dir)
  )

  summary_table <- results |>
    group_by(model) |>
    summarise(
      mean_crps = mean(crps),
      mean_interval_score = mean(interval_score),
      .groups = "drop"
    )

  cat("\n===", country_label, "===\n")
  print(summary_table)

  write_csv(results, file.path(dirname(out_file), paste0(basename(out_file), "_by_observation.csv")))
  write_csv(summary_table, file.path(dirname(out_file), paste0(basename(out_file), "_summary.csv")))

  summary_table
}

# ============================================================================
# CONFIGURATION — adjust paths to the GP run directories to score
# ============================================================================
topdir <- here::here()

tcd_scores <- score_country(
  country_label = "Chad",
  gp_run_dir = file.path(topdir, "output", "gaussian_process", "TCD_gp_with_features_ridge_centered_20261026_132935"),
  br_dir = file.path(topdir, "output", "data", "tcd_bayesian_ridge"),
  br_prefix = "tcd",
  mlp_dir = file.path(topdir, "output", "data", "tcd_mlp"),
  xgb_dir = file.path(topdir, "output", "data", "tcd_xg_boost"),
  out_file = file.path(topdir, "output", "tables", "tcd_proper_scoring_rules")
)

nga_scores <- score_country(
  country_label = "Nigeria",
  gp_run_dir = file.path(topdir, "output", "gaussian_process", "NGA_gp_with_features_normal_centered_20260217_155435"),
  br_dir = file.path(topdir, "output", "data", "nga_bayesian_ridge"),
  br_prefix = "nga",
  mlp_dir = file.path(topdir, "output", "data", "nga_mlp"),
  xgb_dir = file.path(topdir, "output", "data", "nga_xg_boost"),
  out_file = file.path(topdir, "output", "tables", "nga_proper_scoring_rules")
)
