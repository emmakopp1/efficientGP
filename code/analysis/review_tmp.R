# ============================================================================
# Review: overlap sensitivity analysis vs reference GP model (Nigeria)
# ============================================================================
# Compares the reference GP-with-features model (normal centered prior) to
# the sensitivity-analysis variant that adds a region-specific overlap error
# term (tau_ov^2 * R_ov) accounting for the dependence induced by the 90-day
# rolling survey windows.
#
# Reference run:    NGA_gp_with_features_normal_centered_20260217_155435
# Sensitivity run:   NGA_gp_with_features_normal_centered_sensitive_analysis_20261028_100637
#
# Produces, in output/figure/, all prefixed NGA_ridge_sensitive_analysis:
#   - predicted time series overlaid for both models (by CV fold x region)
#   - residuals for both models (by CV fold x region)
#   - metric comparison table (RMSE, MAE, Coverage 95, Length IC 95, CRPS,
#     Interval Score 95), following code/analysis/compute_prediction_metrics.R
# ============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(scoringRules)
  library(here)
})

topdir <- here::here()

path_ref <- file.path(topdir, "output", "gaussian_process", "NGA_gp_with_features_normal_centered_20260217_155435")
path_sens <- file.path(topdir, "output", "gaussian_process", "NGA_gp_with_features_normal_centered_sensitive_analysis_20261028_113634")

n_folds <- 5

# ============================================================================
# STEP 1: Load predictions (ppc_summary) and draws (ppc_samples) for both runs
# ============================================================================

load_gp_summary <- function(run_dir, model_label) {
  diag_dir <- file.path(run_dir, "prior_posterior_diagnostics")
  map_dfr(1:n_folds, function(i) {
    read_csv(file.path(diag_dir, sprintf("ppc_summary_cv%d.csv", i)), show_col_types = FALSE) %>%
      mutate(cv_ind = i, model = model_label)
  })
}

df_ref <- load_gp_summary(path_ref, "Reference")
df_sens <- load_gp_summary(path_sens, "Overlap sensitivity")

df_gp <- bind_rows(df_ref, df_sens) %>%
  rename(
    y_hat = y_pred_mean,
    ci_low = `y_pred_ci_2.5`,
    ci_high = `y_pred_ci_97.5`
  )

# ============================================================================
# STEP 2: Overlay predicted time series (both models + truth) by CV fold x region
# ============================================================================

df_true <- df_gp %>%
  distinct(Datetime, adm1_name, cv_ind, y_true) %>%
  transmute(
    Datetime, adm1_name, cv_ind,
    model = "True",
    cv_ind_adm1_name = paste0("CV ", cv_ind, ": ", adm1_name),
    poor_fcs = y_true,
    ci_low = NA_real_,
    ci_high = NA_real_
  )

df_pred <- df_gp %>%
  transmute(
    Datetime, adm1_name, cv_ind, model,
    cv_ind_adm1_name = paste0("CV ", cv_ind, ": ", adm1_name),
    poor_fcs = y_hat,
    ci_low, ci_high
  )

df_all <- bind_rows(df_pred, df_true) %>%
  mutate(
    model = factor(
      model,
      levels = c("True", "Reference", "Overlap sensitivity"),
      labels = c("True", "GP (reference)", "GP (overlap sensitivity)")
    )
  )

model_cols <- c(
  "GP (reference)" = "#1B7837",
  "GP (overlap sensitivity)" = "#D73027"
)

gg_ts <- ggplot(df_all, aes(
  x = Datetime, y = poor_fcs,
  color = model, linetype = model
)) +
  geom_ribbon(
    data = df_all %>% filter(model != "True"),
    aes(ymin = ci_low, ymax = ci_high, fill = model),
    alpha = 0.25,
    color = NA
  ) +
  geom_line(
    data = df_all %>% filter(model != "True"),
    linewidth = 0.6
  ) +
  geom_line(
    data = df_all %>% filter(model == "True"),
    linewidth = 0.7
  ) +
  facet_wrap(~cv_ind_adm1_name, ncol = 4) +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1)
  ) +
  scale_color_manual(
    values = c("True" = "black", model_cols)
  ) +
  scale_fill_manual(values = model_cols) +
  scale_linetype_manual(
    values = c(
      "True" = "dotted",
      "GP (reference)" = "solid",
      "GP (overlap sensitivity)" = "solid"
    )
  ) +
  theme_minimal(base_size = 10) +
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.title = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 6.5),
    axis.text.y = element_text(angle = 45, hjust = 1, size = 6.5),
    strip.text = element_text(size = 7.5),
    strip.background = element_blank()
  ) +
  labs(
    x = NULL,
    y = "Prevalence of households with insufficient food consumption"
  ) +
  guides(fill = "none")

ggsave(
  filename = file.path(topdir, "output", "figure", "NGA_ridge_sensitive_analysis_predictions_by_adm1.png"),
  plot = gg_ts, width = 6, height = 6, units = "in", dpi = 300, bg = "white"
)

# ============================================================================
# STEP 3: Residual plots for both models, by CV fold x region
# ============================================================================

df_resid <- df_gp %>%
  mutate(
    residual = y_hat - y_true,
    cv_ind_adm1_name = paste0("CV ", cv_ind, ": ", adm1_name),
    model = factor(
      model,
      levels = c("Reference", "Overlap sensitivity"),
      labels = c("GP (reference)", "GP (overlap sensitivity)")
    )
  )

gg_resid_ts <- ggplot(df_resid, aes(x = Datetime, y = residual, color = model)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_line(linewidth = 0.6) +
  facet_wrap(~cv_ind_adm1_name, ncol = 4) +
  scale_color_manual(values = model_cols) +
  theme_minimal(base_size = 10) +
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.title = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 6.5),
    axis.text.y = element_text(angle = 45, hjust = 1, size = 6.5),
    strip.text = element_text(size = 7.5),
    strip.background = element_blank()
  ) +
  labs(x = NULL, y = "Residual (predicted - true)")

ggsave(
  filename = file.path(topdir, "output", "figure", "NGA_ridge_sensitive_analysis_residuals_by_adm1.png"),
  plot = gg_resid_ts, width = 6, height = 6, units = "in", dpi = 300, bg = "white"
)

# ============================================================================
# STEP 4: Metrics comparison (RMSE, MAE, Coverage 95, Length IC 95, CRPS,
# Interval Score 95), following code/analysis/compute_prediction_metrics.R
# ============================================================================

interval_score <- function(y, lower, upper, alpha = 0.05) {
  (upper - lower) +
    (2 / alpha) * (lower - y) * (y < lower) +
    (2 / alpha) * (y - upper) * (y > upper)
}

crps_from_draws <- function(y_true, draws) {
  crps_sample(y = y_true, dat = draws)
}

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
      y_true = summary$y_true,
      y_hat = summary$y_pred_mean,
      ci_low = summary$y_pred_ci_2.5,
      ci_high = summary$y_pred_ci_97.5,
      crps = crps_from_draws(summary$y_true, draws)
    )
  })
}

summarise_model <- function(df, model_name) {
  df <- df %>% filter(!is.na(y_true), !is.na(y_hat))
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

metrics_comparison <- bind_rows(
  summarise_model(metrics_gp(path_ref), "GP (reference)"),
  summarise_model(metrics_gp(path_sens), "GP (overlap sensitivity)")
)

cat("\n==== Metric comparison: reference vs overlap sensitivity analysis ====\n")
print(metrics_comparison %>% mutate(across(where(is.numeric), ~ round(.x, 4))), n = Inf)

write_csv(
  metrics_comparison,
  file.path(topdir, "output", "tables", "NGA_ridge_sensitive_analysis_metrics_comparison.csv")
)

cat("\nSaved figures to output/figure/ with prefix NGA_ridge_sensitive_analysis\n")

# ============================================================================
# Review: overlap sensitivity analysis vs reference GP model (Chad)
# ============================================================================
# Same comparison as above, for Chad: reference GP-with-features model
# (ridge centered prior) vs the sensitivity-analysis variant adding the
# region-specific overlap error term (tau_ov^2 * R_ov).
#
# Reference run:    TCD_gp_with_features_ridge_centered_20261102_182146
# Sensitivity run:   TCD_gp_with_features_ridge_centered_sensitive_analysis_20261102_185709
#
# Chad has 6 CV folds (0-5): fold 0 ("fixed") always stays in the train set
# and is never used as a held-out test fold (see STEP 8 in
# code/analysis/visualise_tcd_result.R), so, mirroring
# code/analysis/compute_prediction_metrics.R, fold 0 is excluded from the
# metric comparison but kept in the time series / residual plots (as
# code/analysis/visualise_tcd_result.R does), with padding panels so every
# CV fold row keeps a consistent 4-column layout.
#
# Produces, in output/figure/, all prefixed TCD_ridge_sensitive_analysis:
#   - predicted time series overlaid for both models (by CV fold x region)
#   - residuals for both models (by CV fold x region)
#   - metric comparison table (RMSE, MAE, Coverage 95, Length IC 95, CRPS,
#     Interval Score 95), following code/analysis/compute_prediction_metrics.R
# ============================================================================

path_ref_tcd <- file.path(topdir, "output", "gaussian_process", "TCD_gp_with_features_ridge_centered_20261102_182146")
path_sens_tcd <- file.path(topdir, "output", "gaussian_process", "TCD_gp_with_features_ridge_centered_sensitive_analysis_20261102_185709")

n_folds_tcd <- 5 # folds 1-5 (plus fold 0, handled separately for display)

# ---- STEP 1: load predictions (ppc_summary) for both runs, folds 0-5 ----
load_gp_summary_tcd <- function(run_dir, model_label) {
  diag_dir <- file.path(run_dir, "prior_posterior_diagnostics")
  map_dfr(0:n_folds_tcd, function(i) {
    read_csv(file.path(diag_dir, sprintf("ppc_summary_cv%d.csv", i)), show_col_types = FALSE) %>%
      mutate(cv_ind = i, model = model_label)
  })
}

df_ref_tcd <- load_gp_summary_tcd(path_ref_tcd, "Reference")
df_sens_tcd <- load_gp_summary_tcd(path_sens_tcd, "Overlap sensitivity")

df_gp_tcd <- bind_rows(df_ref_tcd, df_sens_tcd) %>%
  rename(
    y_hat = y_pred_mean,
    ci_low = `y_pred_ci_2.5`,
    ci_high = `y_pred_ci_97.5`
  )

# ---- STEP 2: overlay predicted time series, with padding panels (4 cols) ----
df_true_tcd <- df_gp_tcd %>%
  distinct(Datetime, adm1_name, cv_ind, y_true) %>%
  transmute(
    Datetime, adm1_name, cv_ind,
    model = "True",
    cv_ind_adm1_name = paste0("CV ", cv_ind, ": ", adm1_name),
    poor_fcs = y_true,
    ci_low = NA_real_,
    ci_high = NA_real_
  )

df_pred_tcd <- df_gp_tcd %>%
  transmute(
    Datetime, adm1_name, cv_ind, model,
    cv_ind_adm1_name = paste0("CV ", cv_ind, ": ", adm1_name),
    poor_fcs = y_hat,
    ci_low, ci_high
  )

df_all_tcd <- bind_rows(df_pred_tcd, df_true_tcd) %>%
  mutate(
    cv_ind = as.integer(cv_ind),
    model = factor(
      model,
      levels = c("True", "Reference", "Overlap sensitivity"),
      labels = c("True", "GP (reference)", "GP (overlap sensitivity)")
    )
  )

# Padding panels so every CV fold row keeps exactly 4 columns, as in
# code/analysis/visualise_tcd_result.R
panel_levels_tcd <- df_all_tcd %>%
  distinct(cv_ind, adm1_name, cv_ind_adm1_name) %>%
  arrange(cv_ind, adm1_name) %>%
  group_by(cv_ind) %>%
  summarise(
    lev = {
      x <- cv_ind_adm1_name
      k <- length(x)
      if (k < 4) {
        pads <- paste0("CV ", unique(cv_ind), ": __pad", seq_len(4 - k))
        c(x, pads)
      } else {
        x
      }
    },
    .groups = "drop"
  ) %>%
  pull(lev) %>%
  unlist()

df_all_tcd <- df_all_tcd %>%
  mutate(cv_ind_adm1_name = factor(cv_ind_adm1_name, levels = panel_levels_tcd))

lab_hide_pad_tcd <- function(x) ifelse(stringr::str_detect(x, "__pad"), "", x)

# Fold 0 ("fixed") is never a held-out test fold; exclude it from display,
# keeping panel_levels_tcd intact for anything else that needs it
panel_levels_tcd_no0 <- panel_levels_tcd[!stringr::str_detect(panel_levels_tcd, "^CV 0:")]

df_all_tcd_plot <- df_all_tcd %>%
  filter(cv_ind != 0) %>%
  mutate(cv_ind_adm1_name = factor(as.character(cv_ind_adm1_name), levels = panel_levels_tcd_no0))

gg_ts_tcd <- ggplot(df_all_tcd_plot, aes(
  x = Datetime, y = poor_fcs,
  color = model, linetype = model
)) +
  geom_ribbon(
    data = df_all_tcd_plot %>% filter(model != "True"),
    aes(ymin = ci_low, ymax = ci_high, fill = model),
    alpha = 0.25,
    color = NA
  ) +
  geom_line(
    data = df_all_tcd_plot %>% filter(model != "True"),
    linewidth = 0.6
  ) +
  geom_line(
    data = df_all_tcd_plot %>% filter(model == "True"),
    linewidth = 0.7
  ) +
  facet_wrap(
    ~cv_ind_adm1_name,
    ncol = 4,
    drop = FALSE,
    labeller = as_labeller(lab_hide_pad_tcd)
  ) +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1)
  ) +
  scale_color_manual(
    values = c("True" = "black", model_cols)
  ) +
  scale_fill_manual(values = model_cols) +
  scale_linetype_manual(
    values = c(
      "True" = "dotted",
      "GP (reference)" = "solid",
      "GP (overlap sensitivity)" = "solid"
    )
  ) +
  theme_minimal(base_size = 10) +
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.title = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 6.5),
    axis.text.y = element_text(angle = 45, hjust = 1, size = 6.5),
    strip.text = element_text(size = 7.5),
    strip.background = element_blank()
  ) +
  labs(
    x = NULL,
    y = "Prevalence of households with insufficient food consumption"
  ) +
  guides(fill = "none")

ggsave(
  filename = file.path(topdir, "output", "figure", "TCD_ridge_sensitive_analysis_predictions_by_adm1.png"),
  plot = gg_ts_tcd, width = 6, height = 7, units = "in", dpi = 300, bg = "white"
)

# ---- STEP 3: residual plots for both models, with padding panels ----
# Fold 0 ("fixed") is never a held-out test fold; excluded from display
df_resid_tcd <- df_gp_tcd %>%
  mutate(
    cv_ind = as.integer(cv_ind),
    residual = y_hat - y_true,
    cv_ind_adm1_name = paste0("CV ", cv_ind, ": ", adm1_name),
    model = factor(
      model,
      levels = c("Reference", "Overlap sensitivity"),
      labels = c("GP (reference)", "GP (overlap sensitivity)")
    )
  ) %>%
  filter(cv_ind != 0) %>%
  mutate(cv_ind_adm1_name = factor(cv_ind_adm1_name, levels = panel_levels_tcd_no0))

gg_resid_ts_tcd <- ggplot(df_resid_tcd, aes(x = Datetime, y = residual, color = model)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_line(linewidth = 0.6) +
  facet_wrap(
    ~cv_ind_adm1_name,
    ncol = 4,
    drop = FALSE,
    labeller = as_labeller(lab_hide_pad_tcd)
  ) +
  scale_color_manual(values = model_cols) +
  theme_minimal(base_size = 10) +
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.title = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 6.5),
    axis.text.y = element_text(angle = 45, hjust = 1, size = 6.5),
    strip.text = element_text(size = 7.5),
    strip.background = element_blank()
  ) +
  labs(x = NULL, y = "Residual (predicted - true)")

ggsave(
  filename = file.path(topdir, "output", "figure", "TCD_ridge_sensitive_analysis_residuals_by_adm1.png"),
  plot = gg_resid_ts_tcd, width = 6, height = 7, units = "in", dpi = 300, bg = "white"
)

# ---- STEP 4: metrics comparison, fold 0 excluded (never held out) ----
metrics_gp_tcd <- function(run_dir) {
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
      y_true = summary$y_true,
      y_hat = summary$y_pred_mean,
      ci_low = summary$y_pred_ci_2.5,
      ci_high = summary$y_pred_ci_97.5,
      crps = crps_from_draws(summary$y_true, draws)
    )
  })
}

metrics_comparison_tcd <- bind_rows(
  summarise_model(metrics_gp_tcd(path_ref_tcd), "GP (reference)"),
  summarise_model(metrics_gp_tcd(path_sens_tcd), "GP (overlap sensitivity)")
)

cat("\n==== Chad: metric comparison: reference vs overlap sensitivity analysis ====\n")
print(metrics_comparison_tcd %>% mutate(across(where(is.numeric), ~ round(.x, 4))), n = Inf)

write_csv(
  metrics_comparison_tcd,
  file.path(topdir, "output", "tables", "TCD_ridge_sensitive_analysis_metrics_comparison.csv")
)

cat("\nSaved figures to output/figure/ with prefix TCD_ridge_sensitive_analysis\n")
