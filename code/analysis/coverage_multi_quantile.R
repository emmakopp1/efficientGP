# Coverage analysis at 50%, 75%, 90%, 95% for GP models only.
# - If fit_cv*.rds are available: recomputes all quantiles from posterior draws.
# - Otherwise falls back to ppc_summary_all_folds.csv (only 50% and 95% available).
# Applies inverse logit transformation (same as gp_runner_lib.R).

suppressPackageStartupMessages({
  library(dplyr)
  library(purrr)
})

ROOT <- normalizePath(file.path(dirname(commandArgs(trailingOnly = FALSE)[
  grep("--file=", commandArgs(trailingOnly = FALSE))
] |> sub("--file=", "", x = _)), "..", ".."))

gp_dirs <- list(
  "NGA — GP with features" = file.path(ROOT, "output/gaussian_process/NGA_gp_with_features_normal_centered_20260217_155435"),
  "NGA — GP no features"   = file.path(ROOT, "output/gaussian_process/NGA_gp_no_features_20251020_161159"),
  "TCD — GP with features" = file.path(ROOT, "output/gaussian_process/TCD_gp_with_features_ridge_centered_20251216_114735"),
  "TCD — GP no features"   = file.path(ROOT, "output/gaussian_process/TCD_gp_no_prior_20251210_112502")
)

probs <- c(0.50, 0.75, 0.90, 0.95)


# ============================================================================
# From posterior draws (fit_cv*.rds) — all quantiles
# ============================================================================

coverage_from_draws <- function(dir_path, model_name) {
  results <- map(1:5, function(i) {
    fit_path  <- file.path(dir_path, paste0("fit_cv", i, ".rds"))
    test_path <- file.path(dir_path, paste0("data_test_cv", i, ".csv"))
    if (!file.exists(fit_path) || !file.exists(test_path)) return(NULL)

    cat("  Fold", i, "(from draws)...\n")
    fit    <- readRDS(fit_path)
    test   <- read.csv(test_path)
    draws  <- 1 / (1 + exp(-fit$draws("y_new", format = "matrix")[, 1:nrow(test)]))
    y_true <- test$fcs

    map2_dbl((1 - probs) / 2, 1 - (1 - probs) / 2, function(lo, hi) {
      mean(y_true >= apply(draws, 2, quantile, probs = lo) &
           y_true <= apply(draws, 2, quantile, probs = hi), na.rm = TRUE)
    }) |> setNames(paste0("Coverage ", probs * 100, "%")) |> as.list() |> as_tibble()
  })

  bind_rows(compact(results)) |>
    summarise(across(everything(), mean, na.rm = TRUE)) |>
    mutate(Model = model_name, .before = 1)
}


# ============================================================================
# From ppc_summary_all_folds.csv — only 50% and 95%
# ============================================================================

coverage_from_ppc <- function(dir_path, model_name) {
  ppc_path <- file.path(dir_path, "prior_posterior_diagnostics", "ppc_summary_all_folds.csv")
  if (!file.exists(ppc_path)) return(NULL)

  cat("  (from ppc_summary — only 50% and 95% available)\n")
  df <- read.csv(ppc_path) |> filter(cv_fold != 0)

  tibble(
    Model          = model_name,
    `Coverage 50%` = mean(df$y_true >= df$y_pred_q25 & df$y_true <= df$y_pred_q75, na.rm = TRUE),
    `Coverage 75%` = NA_real_,
    `Coverage 90%` = NA_real_,
    `Coverage 95%` = mean(df$y_true >= df$y_pred_ci_2.5 & df$y_true <= df$y_pred_ci_97.5, na.rm = TRUE)
  )
}


# ============================================================================
# Run for all models
# ============================================================================

results <- imap(gp_dirs, function(dir_path, model_name) {
  cat("\n==", model_name, "==\n")
  fit_exists <- file.exists(file.path(dir_path, "fit_cv1.rds"))
  if (fit_exists) {
    coverage_from_draws(dir_path, model_name)
  } else {
    coverage_from_ppc(dir_path, model_name)
  }
})

df_coverage <- bind_rows(results)

cat("\n\n")
cat("============================================================\n")
cat("  GP COVERAGE AT 50% / 75% / 90% / 95%\n")
cat("  (NA = quantile not computable without posterior draws)\n")
cat("============================================================\n")
print(df_coverage |> mutate(across(where(is.numeric), ~ round(.x, 3))), n = Inf)
