# Generates figures for the supplementary material.
# STEP 1: Regression coefficients (posterior mean + 95% CI) per CV fold for NGA and TCD.
# STEP 2: MCMC convergence trace plots for GP hyperparameters.
# STEP 3: Length-scale parameter (rho1) selected by nested CV and Rhat values.

# ============================================================================
# STEP 1: Regression Coefficients Analysis
# ============================================================================

library(tidyverse)
library(scales)
library(purrr)
library(here)
library(forcats)
library(patchwork)
library(magick)
library(ggplot2)
library(png)
library(grid)

# Define paths to GP model outputs
path_nigeria <- file.path(
  "output", "gaussian_process",
  "NGA_gp_with_features_normal_centered_20260217_155435"
)

path_chad <- file.path(
  "output", "gaussian_process",
  "TCD_gp_with_features_ridge_centered_20261102_182146"
)

# Feature name mapping for Nigeria
feature_map_nga <- c(
  # climate
  rainfall_3_months_anomaly_cubic_intp = "Rainfall anomaly",
  ndvi_value_cubic_intp                = "NDVI",
  ndvi_anomaly_cubic_intp              = "NDVI anomaly",

  # economic
  log_currency_exchange                = "log (Currency exchange rate)",
  log_food_inflation                   = "log(Food inflation rate)",
  log_food_inflation_MPI               = "log(Food inflation rate) X MPI",
  headline_inflation_cubic_intp        = "Headline inflation",

  # demographic / static
  MPI                                  = "MPI",
  log_Area                             = "log (Area size)",
  Muslim                               = "Muslim population percentage",

  # conflict
  log_fatalities_explosions            = "log(Fatalities from explosion)",
  log_fatalities_battles               = "log(Fatalities from battles)",
  n_conflicts_Battles_rolsum_90days    = "Number of conflicts (battle)",

  # time indicator
  day_num                              = "Day index"
)

# Feature name mapping for Chad
feature_map_chad <- c(
  
  MPI_intensity = "MPI intensity",
  ndvi_value_cubic_intp = "NDVI",
  food_inflation_cubic_intp = "Food inflation rate",
  currency_exchange_log  = "log(Currency exchange rate)",
  Waterways = "Waterways"
)



# Desired order for Nigeria
desired_order_nga <- c(
  "Rainfall anomaly", "NDVI", "NDVI anomaly",
  "log (Currency exchange rate)",
  "Headline inflation",
  "log(Food inflation rate)", "log(Food inflation rate) X MPI",
  "MPI", "log (Area size)", "Muslim population percentage",
  "log(Fatalities from explosion)", "log(Fatalities from battles)",
  "Number of conflicts (battle)",
  "Day index"
)

# Desired order for Chad
desired_order_chad <- c(
  "MPI intensity", "NDVI", "log(Currency exchange rate)",
  "Food inflation rate", "Waterways"
)

# Load coefficiants for all cv folds
df_coef_nga <- map_dfr(1:5, function(i) {
  path_i <- file.path(
    path_nigeria,
    "prior_posterior_diagnostics",
    sprintf("beta_summary_cv%d.csv", i)
  )

  read_csv(path_i, show_col_types = FALSE) %>%
    mutate(cv_fold = i)  # Convert to 1-6 for display
}) %>%
  mutate(
    feature_clean = unname(feature_map_nga[feature]),
    feature_clean = factor(feature_clean, levels = desired_order_nga)
  ) %>%
  arrange(cv_fold, feature_clean)

df_coef_chad <- map_dfr(0:5, function(i) {
  path_i <- file.path(
    path_chad,
    "prior_posterior_diagnostics",
    sprintf("beta_summary_cv%d.csv", i)
  )

  read_csv(path_i, show_col_types = FALSE) %>%
    mutate(cv_fold = i)  # Convert to 1-6 for display
}) %>%
  filter(cv_fold != 0) %>%  # fold 0 is never a held-out test fold
  mutate(
    feature_clean = unname(feature_map_chad[feature]),
    feature_clean = factor(feature_clean, levels = desired_order_chad)
  ) %>%
  filter(!is.na(feature_clean)) %>%  # Remove features not in the mapping
  arrange(cv_fold, feature_clean)


# Create plots
plot_coefficients <- function(df_coef, country_name, show_values = TRUE) {
  # Calculate data range
  min_val <- min(df_coef$ci_2.5, na.rm = TRUE)
  max_val <- max(df_coef$ci_97.5, na.rm = TRUE)
  data_range <- max_val - min_val

  # Calculate right limit for text placement - ensure enough space
  if (show_values) {
    right_limit <- max_val + data_range * 1.2
  } else {
    right_limit <- max_val * 1.1
  }

  p <- df_coef %>%
    mutate(
      feature_clean = fct_rev(fct_inorder(feature_clean)),
      # Create formatted label
      label_ci = sprintf(
        "%6.3f [%6.3f, %6.3f]",
        posterior_mean, ci_2.5, ci_97.5
      )
    ) %>%
    ggplot(aes(x = feature_clean, y = posterior_mean, fill = includes_zero)) +
    geom_col(width = 0.7) +
    geom_errorbar(
      aes(ymin = ci_2.5, ymax = ci_97.5),
      width = 0.3,
      linewidth = 0.4,
      alpha = 0.6
    ) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray40", linewidth = 0.5) +
    facet_wrap(~cv_fold, ncol = 3, labeller = labeller(cv_fold = function(x) paste("Fold", x))) +
    coord_flip() +
    scale_y_continuous(
      limits = c(min_val, right_limit),
      breaks = seq(-1, 1, by = 0.25)
    ) +
    scale_fill_manual(
      values = c("TRUE" = "#2C7FB8", "FALSE" = "#D73027"),
      guide = "none"
    ) +
    labs(
      x = NULL,
      y = "Posterior mean (95% CI)",
      title = paste(country_name, "- Regression Coefficients by CV Fold")
    ) +
    theme_minimal(base_size = 11) +
    theme(
      axis.text.y = element_text(hjust = 0, size = 9),
      axis.text.x = element_text(size = 8.5, angle = 0),
      strip.text = element_text(size = 10, face = "bold"),
      strip.background = element_rect(fill = "grey95", color = NA),
      plot.title = element_text(hjust = 0.5, size = 13, face = "bold"),
      panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      panel.spacing = unit(1, "lines")
    )

  # Add text labels if requested
  if (show_values) {
    p <- p +
      geom_text(
        aes(y = right_limit * 0.98, label = label_ci),
        hjust = 1,
        size = 2.8,
        family = "mono"
      )
  }

  return(p)
}


# Nigeria Plot with values
gg_nga_with_values <- plot_coefficients(
  df_coef_nga,
  "Nigeria",
  show_values = TRUE
)
gg_nga_with_values

ggsave(
  filename = file.path("output", "figure", "NGA_coef_all_cv_with_values.png"),
  plot = gg_nga_with_values,
  width = 14,
  height = 6,
  units = "in",
  dpi = 300,
  bg = "white"
)

# Chad Plot with values
gg_chad_with_values <- plot_coefficients(
  df_coef_chad,
  "Chad",
  show_values = TRUE
)
gg_chad_with_values

ggsave(
  filename = file.path("output", "figure", "CHAD_coef_all_cv_with_values.png"),
  plot = gg_chad_with_values,
  width = 14,
  height = 3,
  units = "in",
  dpi = 300,
  bg = "white"
)


# ============================================================================
# STEP 2: MCMC Convergence Diagnostics - Combined Trace Plot Figures
# ============================================================================

conv_path_nigeria <- file.path(path_nigeria, "convergence_diagnostics")
conv_path_chad <- file.path(path_chad, "convergence_diagnostics")

hyper_files_nigeria <- file.path(conv_path_nigeria, sprintf("trace_plot_cv%d.png", 1:5))
hyper_files_chad    <- file.path(conv_path_chad, sprintf("trace_plot_cv%d.png", 1:5))

# ============================================================================
# Figure 1: trace_plot_cv*.png  (GP hyperparameters chains)
# ============================================================================

# Nigeria
imgs <- image_read(hyper_files_nigeria)

img_width  <- image_info(imgs[1])$width
img_height <- image_info(imgs[1])$height

row1 <- image_append(imgs[1:2])
row2 <- image_append(imgs[3:4])
row3 <- image_append(c(imgs[5], image_blank(img_width, img_height, "white")))

fig_hyper_nigeria <- image_append(c(row1, row2, row3), stack = TRUE)

image_write(
  fig_hyper_nigeria,
  path = file.path("output", "figure", "NGA_trace_hyperparamers_cv.png"),
  format = "png"
)

# Chad
imgs <- image_read(hyper_files_chad)

img_width  <- image_info(imgs[1])$width
img_height <- image_info(imgs[1])$height

row1 <- image_append(imgs[1:2])
row2 <- image_append(imgs[3:4])
row3 <- image_append(c(imgs[5], image_blank(img_width, img_height, "white")))

fig_hyper_chad <- image_append(c(row1, row2, row3), stack = TRUE)

image_write(
  fig_hyper_chad,
  path = file.path("output", "figure", "TCD_trace_hyperparamers_cv.png"),
  format = "png"
)


# ============================================================================
# STEP 3: Scale parameter length cross validation and Rhat value - Figures
# ============================================================================

# rhat statistic paths
rhat_path_nigeria <- file.path(path_nigeria, "final_cv_summary.csv")
rhat_path_chad <- file.path(path_chad, "final_cv_summary.csv")

# length scale parameter cross validation value
rho_res_path_nigeria <- file.path(
  "output", "gaussian_process",
  "NGA_gp_with_features_normal_20260216_154910","final_cv_summary.csv"
)

rho_res_path_chad <- file.path(
  "output", "gaussian_process",
  "TCD_gp_with_features_ridge_centered_20261026_181018","final_cv_summary.csv"
)


# length scale parameter 
# For Nigeria 
rho_cv_nigeria <- read_csv(rho_res_path_nigeria) |> 
  filter(converged == T) |>
  group_by(cv_fold_i, rho1) |> 
  summarise(mean_rMSE = mean(rMSE), 
            sd_rMSE = sd(rMSE),
            .groups = 'drop') |>
  group_by(cv_fold_i) |> 
  slice_min(mean_rMSE, n = 1) |> 
  ungroup() |>
  mutate(mean_rMSE = round(mean_rMSE,3)) |> 
  select(-sd_rMSE) |>
  rename(cv_fold = cv_fold_i, 
         RMSE = mean_rMSE) |> 
  mutate(country = "Nigeria")

# For Chad 
rho_cv_chad <- read_csv(rho_res_path_chad) |> 
  filter(converged == T) |>
  group_by(cv_fold_i, rho1) |> 
  summarise(mean_rMSE = mean(rMSE), 
            sd_rMSE = sd(rMSE),
            .groups = 'drop') |>
  group_by(cv_fold_i) |> 
  slice_min(mean_rMSE, n = 1) |> 
  ungroup() |>
  mutate(mean_rMSE = round(mean_rMSE,3)) |> 
  select(-sd_rMSE) |>
  rename(cv_fold = cv_fold_i, 
         RMSE = mean_rMSE) |> 
  mutate(country = "Chad")


rho_cv <- bind_rows(rho_cv_nigeria, rho_cv_chad)


# Rhat
# Fold 0 is always in train and never a held-out test fold for either
# country, so it is excluded here (reported as "--" in the table).
options(pillar.sigfig = 5)
rhat <- bind_rows(
  read_csv(rhat_path_nigeria) |> select(cv_fold, max_rhat) |> mutate(country = "Nigeria"),
  read_csv(rhat_path_chad)    |> select(cv_fold, max_rhat) |> mutate(country = "Chad")
) |>
  filter(cv_fold != 0) |>
  select(country, cv_fold, max_rhat)

# ESS (bulk), minimum across all model parameters, per fold.
# Read from convergence_summary_cv<fold>.csv (one file per fold), since
# final_cv_summary.csv only carries max_rhat, not ESS.
read_min_ess_bulk <- function(path, country_label) {
  diag_dir <- file.path(path, "convergence_diagnostics")
  list.files(diag_dir, pattern = "^convergence_summary_cv\\d+\\.csv$", full.names = TRUE) |>
    map_dfr(read_csv, show_col_types = FALSE) |>
    mutate(country = country_label) |>
    select(country, cv_fold, min_ess_bulk)
}

ess <- bind_rows(
  read_min_ess_bulk(path_nigeria, "Nigeria"),
  read_min_ess_bulk(path_chad, "Chad")
) |>
  filter(cv_fold != 0)

# Join rhat, ess, and rho
rho_rhat <- rho_cv |>
  left_join(rhat, by = c("country", "cv_fold")) |>
  left_join(ess, by = c("country", "cv_fold"))

rho_rhat

# ============================================================================
# Max Rhat / min ESS (bulk) table, by fold and country (tab:res_cv)
# ============================================================================
# Fold 0 is excluded (always in train, never a held-out test fold for
# either country) and reported as "--".
res_cv_wide <- rhat |>
  rename(value_rhat = max_rhat) |>
  left_join(ess, by = c("country", "cv_fold")) |>
  pivot_wider(
    id_cols = cv_fold,
    names_from = country,
    values_from = c(value_rhat, min_ess_bulk),
    names_glue = "{country}_{.value}"
  ) |>
  arrange(cv_fold)

fmt_rhat <- function(x) if_else(is.na(x), "--", sprintf("%.4f", x))
fmt_ess <- function(x) if_else(is.na(x), "--", sprintf("%.0f", x))

res_cv_rows <- res_cv_wide |>
  transmute(
    row = paste(
      cv_fold,
      fmt_rhat(Nigeria_value_rhat), fmt_ess(Nigeria_min_ess_bulk),
      fmt_rhat(Chad_value_rhat), fmt_ess(Chad_min_ess_bulk),
      sep = " & "
    )
  ) |>
  pull(row) |>
  paste(collapse = " \\\\\n    ")

res_cv_tex <- paste0(
  "\\begin{table}[t!]\n",
  "    \\centering\n",
  "\\caption{Maximum $\\hat{R}$ and minimum effective sample size (ESS, bulk) from MCMC sampling for each fold in Nigeria and Chad, evaluated across all model parameters (Gaussian process hyperparameters and regression coefficients). $\\hat{R}$ values close to 1 and ESS well above 400 indicate satisfactory convergence.}\n",
  "    \\label{tab:res_cv}\n",
  "    \\begin{tabular}{lcccc}\n",
  "    \\toprule\n",
  "    & \\multicolumn{2}{c}{Nigeria} & \\multicolumn{2}{c}{Chad} \\\\\n",
  "    Fold & Max $\\hat{R}$ & Min ESS & Max $\\hat{R}$ & Min ESS \\\\\n",
  "    \\midrule\n",
  "    ", res_cv_rows, " \\\\\n",
  "    \\bottomrule\n",
  "    \\end{tabular}\n",
  "\\end{table}\n"
)

res_cv_path <- file.path("output", "tables", "res_cv.tex")
writeLines(res_cv_tex, res_cv_path)
cat("Saved", res_cv_path, "\n")
cat(res_cv_tex)


# ============================================================================
# STEP 4: Overlap sensitivity analysis vs reference GP model (NGA and TCD)
# ============================================================================
# Compares the reference GP-with-features model to the sensitivity-analysis
# variant that adds a region-specific overlap error term (tau_ov^2 * R_ov)
# accounting for the dependence induced by the 90-day rolling survey windows.
#
# Produces, in output/figure/, prefixed {NGA,TCD}_ridge_sensitive_analysis:
#   - predicted time series overlaid for both models (by CV fold x region)
#   - residuals for both models (by CV fold x region)
#   - metric comparison table (RMSE, MAE, Coverage 95, Length IC 95, CRPS,
#     Interval Score 95), following code/analysis/compute_prediction_metrics.R
# ============================================================================

library(scoringRules)

path_sens_nigeria <- file.path(
  "output", "gaussian_process",
  "NGA_gp_with_features_normal_centered_sensitive_analysis_20261028_113634"
)
path_sens_chad <- file.path(
  "output", "gaussian_process",
  "TCD_gp_with_features_ridge_centered_sensitive_analysis_20261102_185709"
)

interval_score <- function(y, lower, upper, alpha = 0.05) {
  (upper - lower) +
    (2 / alpha) * (lower - y) * (y < lower) +
    (2 / alpha) * (y - upper) * (y > upper)
}

crps_from_draws <- function(y_true, draws) {
  crps_sample(y = y_true, dat = draws)
}

model_cols_sens <- c(
  "GP (reference)" = "#1B7837",
  "GP (overlap sensitivity)" = "#D73027"
)

# has_fold0: Chad has a fold 0 ("fixed") that is always in train and never a
# held-out test fold; Nigeria has no fold 0. When present, fold 0 is loaded
# and shown in the time series / residual plots (with padding panels to keep
# a consistent 4-column layout, as in code/analysis/visualise_tcd_result.R)
# but excluded from the metrics comparison.
run_sensitivity_analysis <- function(country_label, country_prefix,
                                      path_ref, path_sens, has_fold0 = FALSE) {
  diag_ref <- file.path(path_ref, "prior_posterior_diagnostics")
  diag_sens <- file.path(path_sens, "prior_posterior_diagnostics")
  first_fold <- if (has_fold0) 0 else 1

  load_gp_summary <- function(diag_dir, model_label) {
    map_dfr(first_fold:5, function(i) {
      read_csv(file.path(diag_dir, sprintf("ppc_summary_cv%d.csv", i)), show_col_types = FALSE) %>%
        mutate(cv_ind = i, model = model_label)
    })
  }

  df_gp <- bind_rows(
    load_gp_summary(diag_ref, "Reference"),
    load_gp_summary(diag_sens, "Overlap sensitivity")
  ) %>%
    rename(y_hat = y_pred_mean, ci_low = `y_pred_ci_2.5`, ci_high = `y_pred_ci_97.5`)

  # ---- Predicted time series, by CV fold x region ----
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
      cv_ind = as.integer(cv_ind),
      model = factor(
        model,
        levels = c("True", "Reference", "Overlap sensitivity"),
        labels = c("True", "GP (reference)", "GP (overlap sensitivity)")
      )
    )

  n_panel_cols <- 4

  if (has_fold0) {
    # Padding panels so every CV fold row keeps exactly 4 columns
    panel_levels <- df_all %>%
      distinct(cv_ind, adm1_name, cv_ind_adm1_name) %>%
      arrange(cv_ind, adm1_name) %>%
      group_by(cv_ind) %>%
      summarise(
        lev = {
          x <- cv_ind_adm1_name
          k <- length(x)
          if (k < n_panel_cols) {
            pads <- paste0("CV ", unique(cv_ind), ": __pad", seq_len(n_panel_cols - k))
            c(x, pads)
          } else {
            x
          }
        },
        .groups = "drop"
      ) %>%
      pull(lev) %>%
      unlist()

    lab_hide_pad <- function(x) ifelse(stringr::str_detect(x, "__pad"), "", x)
    # Fold 0 is never a held-out test fold; excluded from the plots
    panel_levels_no0 <- panel_levels[!stringr::str_detect(panel_levels, "^CV 0:")]

    df_all <- df_all %>%
      filter(cv_ind != 0) %>%
      mutate(cv_ind_adm1_name = factor(as.character(cv_ind_adm1_name), levels = panel_levels_no0))
    facet_drop <- FALSE
    facet_labeller <- as_labeller(lab_hide_pad)
    plot_height <- 7
  } else {
    facet_drop <- TRUE
    facet_labeller <- "label_value"
    plot_height <- 6
  }

  gg_ts <- ggplot(df_all, aes(x = Datetime, y = poor_fcs, color = model, linetype = model)) +
    geom_ribbon(
      data = df_all %>% filter(model != "True"),
      aes(ymin = ci_low, ymax = ci_high, fill = model),
      alpha = 0.25,
      color = NA
    ) +
    geom_line(data = df_all %>% filter(model != "True"), linewidth = 0.6) +
    geom_line(data = df_all %>% filter(model == "True"), linewidth = 0.7) +
    facet_wrap(~cv_ind_adm1_name, ncol = n_panel_cols, drop = facet_drop, labeller = facet_labeller) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    scale_color_manual(values = c("True" = "black", model_cols_sens)) +
    scale_fill_manual(values = model_cols_sens) +
    scale_linetype_manual(
      values = c("True" = "dotted", "GP (reference)" = "solid", "GP (overlap sensitivity)" = "solid")
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
    labs(x = NULL, y = "Prevalence of households with insufficient food consumption") +
    guides(fill = "none")

  ggsave(
    filename = file.path("output", "figure", paste0(country_prefix, "_ridge_sensitive_analysis_predictions_by_adm1.png")),
    plot = gg_ts, width = 6, height = plot_height, units = "in", dpi = 300, bg = "white"
  )

  # ---- Residuals, by CV fold x region ----
  df_resid <- df_gp %>%
    mutate(
      cv_ind = as.integer(cv_ind),
      residual = y_hat - y_true,
      cv_ind_adm1_name = paste0("CV ", cv_ind, ": ", adm1_name),
      model = factor(
        model,
        levels = c("Reference", "Overlap sensitivity"),
        labels = c("GP (reference)", "GP (overlap sensitivity)")
      )
    )

  if (has_fold0) {
    df_resid <- df_resid %>%
      filter(cv_ind != 0) %>%
      mutate(cv_ind_adm1_name = factor(cv_ind_adm1_name, levels = panel_levels_no0))
  }

  gg_resid <- ggplot(df_resid, aes(x = Datetime, y = residual, color = model)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    geom_line(linewidth = 0.6) +
    facet_wrap(~cv_ind_adm1_name, ncol = n_panel_cols, drop = facet_drop, labeller = facet_labeller) +
    scale_color_manual(values = model_cols_sens) +
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
    filename = file.path("output", "figure", paste0(country_prefix, "_ridge_sensitive_analysis_residuals_by_adm1.png")),
    plot = gg_resid, width = 6, height = plot_height, units = "in", dpi = 300, bg = "white"
  )

  # ---- Metrics comparison (fold 0 excluded: never a held-out test fold) ----
  metrics_gp <- function(path_run) {
    diag_dir <- file.path(path_run, "prior_posterior_diagnostics")
    sample_files <- list.files(diag_dir, pattern = "^ppc_samples_cv\\d+\\.csv$", full.names = TRUE)

    map_dfr(sample_files, function(f) {
      cv_fold <- as.integer(gsub("\\D", "", basename(f)))
      if (cv_fold == 0) return(NULL)

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

  # Matches code/analysis/compute_prediction_metrics.R: metrics are computed
  # per (cv_fold, adm1_name) group, averaged over folds within each region,
  # then averaged over regions (not pooled over all observations directly).
  summarise_model <- function(df, model_name) {
    df <- df %>% filter(!is.na(y_true), !is.na(y_hat))

    per_region_fold <- df %>%
      group_by(cv_fold, adm1_name) %>%
      summarise(
        RMSE = sqrt(mean((y_hat - y_true)^2)),
        MAE = mean(abs(y_hat - y_true)),
        `Coverage 95` = mean(y_true >= ci_low & y_true <= ci_high),
        `Length IC 95` = mean(ci_high - ci_low),
        CRPS = mean(crps),
        `Interval Score 95` = mean(interval_score(y_true, ci_low, ci_high)),
        .groups = "drop"
      )

    per_region_fold %>%
      group_by(adm1_name) %>%
      summarise(across(c(RMSE, MAE, `Coverage 95`, `Length IC 95`, CRPS, `Interval Score 95`), mean), .groups = "drop") %>%
      summarise(across(c(RMSE, MAE, `Coverage 95`, `Length IC 95`, CRPS, `Interval Score 95`), mean)) %>%
      mutate(Model = model_name, .before = 1)
  }

  metrics_comparison <- bind_rows(
    summarise_model(metrics_gp(path_ref), "GP (reference)"),
    summarise_model(metrics_gp(path_sens), "GP (overlap sensitivity)")
  )

  cat("\n====", country_label, ": metric comparison, reference vs overlap sensitivity analysis ====\n")
  print(metrics_comparison %>% mutate(across(where(is.numeric), ~ round(.x, 4))), n = Inf)

  write_csv(
    metrics_comparison,
    file.path("output", "tables", paste0(country_prefix, "_ridge_sensitive_analysis_metrics_comparison.csv"))
  )

  cat("\nSaved figures to output/figure/ with prefix", paste0(country_prefix, "_ridge_sensitive_analysis"), "\n")

  metrics_comparison
}

metrics_sens_nga <- run_sensitivity_analysis(
  "Nigeria", "NGA", path_nigeria, path_sens_nigeria, has_fold0 = FALSE
)

metrics_sens_tcd <- run_sensitivity_analysis(
  "Chad", "TCD", path_chad, path_sens_chad, has_fold0 = TRUE
)

# ============================================================================
# Overlap sensitivity analysis comparison table (tab:prediction_metrics_sens)
# GP reference vs GP overlap sensitivity, for Nigeria and Chad
# ============================================================================
sens_table_rows <- bind_rows(
  metrics_sens_nga %>% mutate(Country = "Nigeria", .before = 1),
  metrics_sens_tcd %>% mutate(Country = "Chad", .before = 1)
) %>%
  mutate(
    Model = recode(Model,
      "Reference" = "GP with covariates",
      "GP (reference)" = "GP with covariates",
      "GP (overlap sensitivity)" = "GP with covariates (overlap sensitivity)"
    )
  )

fmt_row <- function(country, model, rmse, mae, coverage, mil, is_score, crps) {
  sprintf(
    "%s & %s & %.3f & %.3f & %s & %.3f & %.3f & %.3f \\\\",
    country, model, rmse, mae,
    paste0(sprintf("%.1f", coverage * 100), "\\%"),
    mil, is_score, crps
  )
}

sens_rows_tex <- sens_table_rows %>%
  transmute(
    row = fmt_row(
      Country, Model, RMSE, MAE, `Coverage 95`, `Length IC 95`, `Interval Score 95`, CRPS
    )
  ) %>%
  pull(row)

# Blank the Country label on the second row of each pair, as in
# tab:prediction_metrics (Country only printed once per pair of rows)
sens_rows_tex[c(2, 4)] <- sub("^[A-Za-z]+ & ", " & ", sens_rows_tex[c(2, 4)])

sens_table_tex <- paste0(
  "\\begin{table}[t!]\n",
  "    \\centering\n",
  "    \\caption{Overlap sensitivity analysis: metrics for the reference GP model with covariates versus the variant including a region-specific overlap error term, for Nigeria and Chad.}\n",
  "    \\label{tab:prediction_metrics_sens}\n",
  "    \\begin{tabular}{llcccccc}\n",
  "    \\toprule\n",
  "    & & & & \\multicolumn{3}{c}{95\\% \\textcolor{blue}{Predictive Interval}} & \\\\\n",
  "    \\cmidrule(lr){5-7}\n",
  "    Country & Model & RMSE & MAE & Coverage & MIL & Interval Score & CRPS \\\\\n",
  "    \\midrule\n",
  "    ", sens_rows_tex[1], "\n",
  "    ", sens_rows_tex[2], "\n",
  "    \\midrule\n",
  "    ", sens_rows_tex[3], "\n",
  "    ", sens_rows_tex[4], "\n",
  "    \\bottomrule\n",
  "    \\end{tabular}\n",
  "\\end{table}\n"
)

sens_table_path <- file.path("output", "tables", "prediction_metrics_sensitivity_analysis.tex")
writeLines(sens_table_tex, sens_table_path)
cat("Saved", sens_table_path, "\n")
cat(sens_table_tex)



