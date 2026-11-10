# ============================================================================
# Compute prediction metrics (RMSE, MAE, 95% coverage, 95% interval length,
# CRPS, interval score) for all models, for Nigeria and Chad, and save as a
# single combined LaTeX table: output/tables/prediction_metrics_combined.tex.
#
# CRPS and interval score definitions follow
# code/analysis/proper_scoring_rules_definitions.tex:
#   - CRPS: empirical estimator (scoringRules::crps_sample) from model draws
#   - Interval score: IS_alpha at alpha = 0.05 (95% interval)
#
# Per-model draws (added alongside the existing point/interval predictions):
#   - GP:    ppc_samples_cv*.csv (joined to ppc_summary_cv*.csv)
#   - BYM2:  ppc_samples_cv*.csv (joined to ppc_summary_cv*.csv)
#   - MLP:   mc_samples_cv*.csv (joined to predictions_cv*.csv)
#   - XGBoost: bootstrap_preds_cv*.csv (joined to predictions_cv*.csv)
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
      cv_fold = cv_fold,
      adm1_name = summary$adm1_name,
      y_true = summary$y_true,
      y_hat = summary$y_pred_mean,
      ci_low = summary$y_pred_ci_2.5,
      ci_high = summary$y_pred_ci_97.5,
      crps = crps_from_draws(summary$y_true, draws)
    )
  })
}

metrics_bym <- function(run_dir) {
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
      cv_fold = cv_fold,
      adm1_name = summary$adm1_name,
      y_true = summary$y_true,
      y_hat = summary$y_pred_mean,
      ci_low = summary$y_pred_ci_2.5,
      ci_high = summary$y_pred_ci_97.5,
      crps = crps_from_draws(summary$y_true, draws)
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
      cv_fold = fold,
      adm1_name = preds$adm1_name,
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
      cv_fold = fold,
      adm1_name = preds$adm1_name,
      y_true = preds$y_true,
      y_hat = preds$y_hat,
      ci_low = preds$ci_low,
      ci_high = preds$ci_high,
      crps = crps_from_draws(samples$y_true, draws)
    )
  })
}

# ============================================================================
# Summarise one model's per-observation metrics into a single table row.
# Matches the original metrics.py aggregation: metrics are computed per
# (cv_fold, adm1_name) group, averaged over folds within each region, then
# averaged over regions (not pooled over all observations directly).
# ============================================================================
summarise_model <- function(df, model_name) {
  df <- df |> filter(!is.na(y_true), !is.na(y_hat))

  per_region_fold <- df |>
    group_by(cv_fold, adm1_name) |>
    summarise(
      RMSE = sqrt(mean((y_hat - y_true)^2)),
      MAE = mean(abs(y_hat - y_true)),
      `Coverage 95` = mean(y_true >= ci_low & y_true <= ci_high),
      `Length IC 95` = mean(ci_high - ci_low),
      CRPS = mean(crps),
      `Interval Score 95` = mean(interval_score(y_true, ci_low, ci_high)),
      .groups = "drop"
    )

  per_region_fold |>
    group_by(adm1_name) |>
    summarise(across(c(RMSE, MAE, `Coverage 95`, `Length IC 95`, CRPS, `Interval Score 95`), mean), .groups = "drop") |>
    summarise(across(c(RMSE, MAE, `Coverage 95`, `Length IC 95`, CRPS, `Interval Score 95`), mean)) |>
    mutate(Model = model_name, .before = 1)
}

# ============================================================================
# CRPS and Interval Score require draws from an actual predictive
# distribution; MLP (MC Dropout) and XGBoost (bootstrap) only approximate
# model/parameter uncertainty, not the full predictive distribution, so
# their CRPS/Interval Score values are not comparable to the Bayesian
# models' and are displayed as "--" instead.
# ============================================================================
no_distribution_models <- c("MLP", "XG-Boost")
no_distribution_cols <- c("CRPS", "Interval Score 95")

# ============================================================================
# Write the combined Nigeria + Chad LaTeX table (tab:prediction_metrics):
# one row per model, countries grouped with \multirow-free blank-cell rows,
# columns re-ordered as RMSE, MAE, Coverage, MIL, Interval Score, CRPS.
# ============================================================================
model_label <- c(
  "BYM2" = "Bayesian Ridge with BYM2",
  "GP No Features" = "GP without covariates",
  "GP With Features" = "GP with covariates",
  "MLP" = "MLP",
  "XG-Boost" = "XGBoost"
)
model_order <- names(model_label)

write_combined_metrics_table <- function(nga_df, tcd_df, caption, label, out_path) {
  fmt <- function(x) sprintf("%.3f", x)
  fmt_pct <- function(x) {
    pct <- x * 100
    ifelse(
      abs(pct - round(pct)) < 1e-9,
      sprintf("%.0f\\%%", pct),
      sprintf("%.1f\\%%", pct)
    )
  }

  prep_country <- function(df, country_name) {
    df <- df |> mutate(Model = factor(Model, levels = model_order)) |> arrange(Model)
    is_no_dist <- as.character(df$Model) %in% no_distribution_models
    tibble(
      Country = country_name,
      Model = model_label[as.character(df$Model)],
      RMSE = fmt(df$RMSE),
      MAE = fmt(df$MAE),
      Coverage = fmt_pct(df$`Coverage 95`),
      MIL = fmt(df$`Length IC 95`),
      `Interval Score` = if_else(is_no_dist, "--", fmt(df$`Interval Score 95`)),
      CRPS = if_else(is_no_dist, "--", fmt(df$CRPS))
    )
  }

  combined <- bind_rows(
    prep_country(nga_df, "Nigeria"),
    prep_country(tcd_df, "Chad")
  )

  # Blank the Country cell for every row after the first one within each
  # country block, so only one label appears per block (as in the template).
  combined <- combined |>
    group_by(Country) |>
    mutate(Country = if_else(row_number() == 1, Country, "")) |>
    ungroup()

  rows <- combined |>
    unite("row", everything(), sep = " & ") |>
    pull(row) |>
    paste0(" \\\\")

  # Insert a \midrule between the Nigeria and Chad blocks (right before the
  # first row whose Country cell is "Chad")
  chad_start <- which(combined$Country == "Chad")[1]
  rows <- append(rows, "\\midrule", after = chad_start - 1)
  body <- paste(paste0("    ", rows), collapse = "\n")

  tex <- paste0(
    "\\begin{table}[t!]\n",
    "    \\centering\n",
    "    \\caption{", caption, "}\n",
    "    \\label{", label, "}\n",
    "    \\begin{tabular}{llcccccc}\n",
    "    \\toprule\n",
    "    & & & & \\multicolumn{3}{c}{95\\% \\textcolor{blue}{Predictive Interval}} & \\\\\n",
    "    \\cmidrule(lr){5-7}\n",
    "    Country & Model & RMSE & MAE & Coverage & MIL & Interval Score & CRPS \\\\\n",
    "    \\midrule\n",
    body, "\n",
    "    \\bottomrule\n",
    "    \\end{tabular}\n",
    "\\end{table}\n"
  )

  writeLines(tex, out_path)
  cat("Saved", out_path, "\n")
}

# ============================================================================
# Run for one country
# ============================================================================
run_country <- function(country_label, gp_with_features_dir, gp_no_features_dir,
                         bym_dir, mlp_dir, xgb_dir, extra_bym_dirs = NULL) {
  cat("\n==", country_label, "==\n")

  results <- bind_rows(
    summarise_model(metrics_gp(gp_with_features_dir), "GP With Features"),
    summarise_model(metrics_gp(gp_no_features_dir), "GP No Features"),
    summarise_model(metrics_bym(bym_dir), "BYM2"),
    summarise_model(metrics_mlp(mlp_dir), "MLP"),
    summarise_model(metrics_xgboost(xgb_dir), "XG-Boost")
  )

  # Extra BYM2 variants (e.g. ablations), named list of run_dir, model name = list name
  if (!is.null(extra_bym_dirs)) {
    results <- bind_rows(
      results,
      imap(extra_bym_dirs, ~ summarise_model(metrics_bym(.x), .y)) |> bind_rows()
    )
  }

  print(results |> mutate(across(where(is.numeric), ~ round(.x, 3))), n = Inf)

  results
}

# ============================================================================
# CONFIGURATION
# ============================================================================

tcd_metrics <- run_country(
  country_label = "Chad",
  gp_with_features_dir = file.path(topdir, "output", "gaussian_process", "TCD_gp_with_features_ridge_centered_20261102_182146"),
  gp_no_features_dir = file.path(topdir, "output", "gaussian_process", "TCD_gp_no_prior_20261102_192104"),
  bym_dir = file.path(topdir, "output", "bym2", "TCD_bym2_with_features_ridge_20261106_141613"),
  mlp_dir = file.path(topdir, "output", "data", "tcd_mlp"),
  xgb_dir = file.path(topdir, "output", "data", "tcd_xg_boost")
)

nga_metrics <- run_country(
  country_label = "Nigeria",
  gp_with_features_dir = file.path(topdir, "output", "gaussian_process", "NGA_gp_with_features_ridge_centered_20261105_222343"),
  gp_no_features_dir = file.path(topdir, "output", "gaussian_process", "NGA_gp_no_features_20251020_161159"),
  bym_dir = file.path(topdir, "output", "bym2", "NGA_bym2_with_features_ridge_20261106_135255"),
  mlp_dir = file.path(topdir, "output", "data", "nga_mlp"),
  xgb_dir = file.path(topdir, "output", "data", "nga_xg_boost")
)

# Combined Nigeria + Chad table (tab:prediction_metrics)
write_combined_metrics_table(
  nga_df = nga_metrics,
  tcd_df = tcd_metrics,
  caption = paste(
    "Metrics results for Nigeria and Chad across methods.",
    "CRPS and Interval Score are not reported (--) for MLP and XGBoost:",
    "their uncertainty estimates (MC Dropout and bootstrap, respectively)",
    "approximate model/parameter uncertainty only, not the full predictive",
    "distribution, and are therefore not comparable to the Bayesian models'."
  ),
  label = "tab:prediction_metrics",
  out_path = file.path(topdir, "output", "tables", "prediction_metrics_combined.tex")
)
