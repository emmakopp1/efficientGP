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
  "TCD_gp_with_features_ridge_centered_20251216_114735"
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
  headline_inflation_cubic_intp        = "log(Headline inflation)",

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
  "log(Headline inflation)",
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
  "TCD_gp_with_features_normal_20251219_182307","final_cv_summary.csv"
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
options(pillar.sigfig = 5)
rhat <- bind_rows(
  read_csv(rhat_path_nigeria) |> select(cv_fold, max_rhat) |> mutate(country = "Nigeria"),
  read_csv(rhat_path_chad)    |> select(cv_fold, max_rhat) |> mutate(country = "Chad")
) |> 
  select(country, cv_fold, max_rhat)

# Join rhat and rho
rho_rhat <- rho_cv |>
  left_join(rhat, by = c("country", "cv_fold"))

rho_rhat



