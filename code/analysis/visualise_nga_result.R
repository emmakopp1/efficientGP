# ============================================================================
# Nigeria Food Insecurity Prediction Model Comparison
# ============================================================================
# This script compares 4 different models for predicting food insecurity:
# 1. Gaussian Process (GP)
# 2. Bayesian Ridge Regression
# 3. Multi-Layer Perceptron (MLP)
# 4. XGBoost
#
# It generates:
# - Time series plots of predictions vs. true values
# - Performance metrics (RMSE, MAE, coverage, interval scores)
# - Comparison plots by region and cross-validation fold
# ============================================================================

library(tidyverse)
library(scales)
library(purrr)
library(stringr)
library(sf)
library(ggpattern)
library(ggrepel)
library(wesanderson)
library(here)
library(forcats)

# Define paths to model output directories
path_nga <- c(
  file.path(
    "output", "gaussian_process",
    "NGA_gp_with_features_normal_centered_20260217_155435"
  ),
  file.path("output", "data", "nga_bayesian_ridge"),
  file.path("output", "data", "nga_mlp"),
  file.path("output", "data", "nga_xg_boost")
)


# ============================================================================
# STEP 1: Load predictions from all models (5-fold cross-validation)
# ============================================================================

# ---- Bayesian Ridge ----
# Load Bayesian Ridge predictions for each CV fold
path <- path_nga[[2]]
df_bayesian_ridge <- map_dfr(1:5, function(i) {
  read_csv(file.path(path, sprintf("nga_predictions_fold%d.csv", i)), show_col_types = FALSE) %>%
    mutate(
      cv_ind = i,
      model  = "bayesian ridge"
    )
})

# ---- MLP ----
# Load MLP predictions and select relevant columns
path <- path_nga[[3]]

df_mlp <- map_dfr(1:5, function(i) {
  read_csv(file.path(path, sprintf("predictions_cv%d.csv", i)), show_col_types = FALSE) %>%
    select(Datetime, y_hat, y_true, ci_low, ci_high, adm1_name) %>%
    mutate(
      cv_ind = i,
      model  = "mlp"
    )
})

# ---- Gaussian process ----
# Load GP predictions and rename columns to match other models
path <- path_nga[[1]]

df_gp <- map_dfr(1:5, function(i) {
  path_i <- file.path(path, "prior_posterior_diagnostics", sprintf("ppc_summary_cv%d.csv", i))

  read_csv(path_i, show_col_types = FALSE) %>%
    rename(
      y_hat   = y_pred_mean,
      ci_low  = `y_pred_ci_2.5`,
      ci_high = `y_pred_ci_97.5`
    ) %>%
    select(Datetime, y_hat, y_true, ci_low, ci_high, adm1_name) %>%
    mutate(
      cv_ind = i,
      model  = "gaussian process"
    )
})

# ---- XGBoost ----
# Load XGBoost predictions, filter for survey data, and harmonize column names
# Only keep regions that are also in the GP dataset for fair comparison
path <- path_nga[[4]]

df_xg_boost <- map_dfr(1:5, function(i) {
  read_csv(file.path(here(),path, sprintf("predictions_cv%d.csv", i)),
    show_col_types = FALSE
  ) |> 
    #filter(data_type == "SURVEY 90 days", is.na(last_fcs)) |>
    #select(Datetime, fcs_prediction, fcs, fcs_prediction_lower_bound, fcs_prediction_upper_bound, adm1_name) |>
    #rename(
    #  Datetime = date,
    #  y_hat    = fcs_prediction,
    #  y_true   = fcs,
    #  ci_low   = fcs_prediction_lower_bound,
    #  ci_high  = fcs_prediction_upper_bound
    #) %>%
    filter(adm1_name %in% (df_gp |> filter(cv_ind == i) |> pull(adm1_name) |> unique())) |>
    mutate(
      cv_ind = i,
      model  = "xg-boost"
    )
})

# ============================================================================
# STEP 2: Prepare data for visualization
# ============================================================================

# Extract true values from GP dataset (same for all models)
df_true <- df_gp |>
  select(Datetime, adm1_name, y_true, cv_ind) |>
  transmute(
    Datetime,
    adm1_name,
    cv_ind,
    model = "True",
    cv_ind_adm1_name = paste0("CV ", cv_ind, ": ", adm1_name),
    poor_fcs = y_true,
    ci_low = NA_real_,
    ci_high = NA_real_
  )

# Combine all model predictions with harmonized column names
df_pred <- bind_rows(df_gp, df_bayesian_ridge, df_mlp, df_xg_boost) |>
  transmute(
    Datetime,
    adm1_name,
    cv_ind,
    model,
    cv_ind_adm1_name = paste0("CV ", cv_ind, ": ", adm1_name),
    poor_fcs = y_hat,
    ci_low,
    ci_high
  )

# Combine true values and predictions, then set factor levels for consistent ordering
df_all <- bind_rows(df_pred, df_true) |>
  mutate(
    model = factor(
      model,
      levels = c(
        "True",
        "gaussian process",
        "bayesian ridge",
        "mlp",
        "xg-boost"
      ),
      labels = c(
        "True",
        "GP with covariates",
        "Bayesian Ridge",
        "MLP",
        "XGBoost"
      )
    )
  )

# Define color scheme for each model
model_cols <- c(
  "GP with covariates" = "#1B7837",
  "Bayesian Ridge" = "#D73027",
  "MLP" = "#2C7FB8",
  "XGBoost" = "#8E63CE"
)

gg <- ggplot(df_all, aes(
  x = Datetime, y = poor_fcs,
  color = model, linetype = model
)) +

  # Ribbons (prediction models only)
  geom_ribbon(
    data = df_all |> filter(model != "True"),
    aes(ymin = ci_low, ymax = ci_high, fill = model),
    alpha = 0.3,
    color = NA
  ) +

  # Prediction lines first
  geom_line(
    data = df_all |> filter(model != "True"),
    linewidth = 0.6
  ) +

  # TRUE drawn last so it is on top
  geom_line(
    data = df_all |> filter(model == "True"),
    linewidth = 0.7
  ) +
  facet_wrap(~cv_ind_adm1_name, ncol = 4) +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1)
  ) +
  scale_color_manual(
    values = c(
      "True" = "black",
      model_cols
    )
  ) +
  scale_fill_manual(
    values = model_cols
  ) +
  scale_linetype_manual(
    values = c(
      "True" = "dotted",
      "GP with covariates" = "solid",
      "Bayesian Ridge" = "solid",
      "MLP" = "solid",
      "XGBoost" = "solid"
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
gg

# Save the plot
ggsave(
  filename = file.path("output", "figure", "NGA_fcs_prev_all_by_adm1.png"),
  plot = gg, width = 6, height = 6, units = "in", dpi = 300
)

# ============================================================================
# STEP 4: Calculate performance metrics for each model
# ============================================================================

# Significance level for confidence intervals
alpha <- 0.05

# Calculate prediction errors and interval metrics
df_pred_metrics <- bind_rows(df_gp, df_bayesian_ridge, df_mlp, df_xg_boost) |>
  mutate(
    model = factor(
      model,
      levels = c(
        "True",
        "gaussian process",
        "bayesian ridge",
        "mlp",
        "xg-boost"
      ),
      labels = c(
        "True",
        "GP with covariates",
        "Bayesian Ridge",
        "MLP",
        "XGBoost"
      )
    ),
    cv_ind_adm1_name = paste0("CV ", cv_ind, ": ", adm1_name),
    error = y_hat - y_true,
    # Check if true value falls within confidence interval
    covered = (y_true >= ci_low & y_true <= ci_high),
    # Width of confidence interval
    int_length = ci_high - ci_low,
    # Interval score: penalizes wide intervals and observations outside the interval
    interval_score = ifelse(
      is.na(ci_low) | is.na(ci_high),
      NA_real_,
      (ci_high - ci_low) +
        (2 / alpha) * pmax(ci_low - y_true, 0) +
        (2 / alpha) * pmax(y_true - ci_high, 0)
    )
  )

# Aggregate metrics by region and CV fold
df_all_metrics <- df_pred_metrics |>
  group_by(model, cv_ind, adm1_name) |>
  summarise(
    rmse = sqrt(mean(error^2, na.rm = TRUE)),  # Root Mean Squared Error
    mae = mean(abs(error), na.rm = TRUE),      # Mean Absolute Error
    coverage = mean(covered, na.rm = TRUE),    # Proportion of true values in CI
    mean_int_length = mean(int_length, na.rm = TRUE),  # Average CI width
    mean_interval_score = mean(interval_score, na.rm = TRUE)  # Average interval score
  )

# Helper function: create wide-format table and identify best-performing model
# Lower values are better for all metrics
make_metric_table <- function(df, metric_col, best_col_name) {
  metric_col <- rlang::ensym(metric_col)

  # Reshape to wide format (one column per model)
  wide <- df |>
    select(cv_ind, adm1_name, model, !!metric_col) |>
    pivot_wider(
      names_from = model,
      values_from = !!metric_col
    )

  model_cols <- setdiff(names(wide), c("cv_ind", "adm1_name"))

  # Identify best model (minimum value) for each region/fold
  wide |>
    rowwise() |>
    mutate(
      !!best_col_name := {
        vals <- c_across(all_of(model_cols))
        # Ignore NAs; pick the model with smallest value
        if (all(is.na(vals))) {
          NA_character_
        } else {
          model_cols[which.min(replace(vals, is.na(vals), Inf))]
        }
      }
    ) |>
    ungroup() |>
    arrange(cv_ind, adm1_name)
}

# Create comparison tables for different metrics
# 1) RMSE table
tab_rmse <- make_metric_table(
  df_all_metrics,
  rmse,
  "best_model_rmse"
)
tab_rmse

# 2) Interval score table
tab_intscore <- make_metric_table(
  df_all_metrics,
  mean_interval_score,
  "best_model_intscore"
)
tab_intscore

# 3) Coverage table
tab_coverage <- make_metric_table(
  df_all_metrics,
  coverage,
  "best_model_coverage"
)
tab_coverage

df_plot_metrics <- df_all_metrics |>
  mutate(cv_ind_adm1_name = paste0("CV ", cv_ind, ": ", adm1_name)) |>
  pivot_longer(
    cols = c(rmse, mae),
    names_to = "metric",
    values_to = "value"
  ) |>
  group_by(cv_ind, adm1_name, metric) |>
  mutate(is_best = value == min(value, na.rm = TRUE)) |>
  ungroup() |>
  mutate(
    metric = recode(metric, rmse = "RMSE", mae = "MAE")
  )

gg <- ggplot(df_plot_metrics |> filter(metric == "RMSE"), aes(x = model, y = value)) +
  geom_col(aes(fill = is_best), width = 0.8, color = "white") +
  # facet_wrap(~ cv_ind_adm1_name, scales = "free_y", ncol = 4) +
  facet_wrap(~cv_ind_adm1_name, ncol = 4) +
  scale_fill_manual(values = c(`TRUE` = "gold", `FALSE` = "grey80")) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 90, hjust = 1),
    strip.background = element_blank()
  ) +
  labs(x = NULL, y = "RMSE")
gg

ggsave(
  filename = file.path("output", "figure", "NGA_RMSE_all_by_adm1.png"),
  plot = gg, width = 6, height = 7.5, units = "in", dpi = 300, bg = "white"
)

# Prepare data for colored plot
df_plot_rmse <- df_all_metrics |>
  mutate(cv_ind_adm1_name = paste0("CV ", cv_ind, ": ", adm1_name)) |>
  group_by(cv_ind, adm1_name) |>
  mutate(is_best = rmse == min(rmse, na.rm = TRUE)) |>
  ungroup()

# Plot 2: Colored bars by model, with transparency for non-best models
gg <- ggplot(
  df_plot_rmse,
  aes(x = model, y = rmse, fill = model, alpha = is_best)
) +
  geom_col(width = 0.8, color = "white") +
  # facet_wrap(~ cv_ind_adm1_name, scales = "free_y", ncol = 4) +
  facet_wrap(~cv_ind_adm1_name, ncol = 4) +
  # Use model-specific colors
  scale_fill_manual(values = model_cols) +
  # Best models are fully opaque, others are transparent
  scale_alpha_manual(
    values = c(`TRUE` = 1, `FALSE` = 0.35),
    guide = "none"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    # axis.text.x = element_text(angle = 90, hjust = 1),
    legend.title = element_blank(),
    axis.text.x = element_blank(),
    strip.background = element_blank(),
    strip.text = element_text(size = 7.5)
  ) +
  labs(
    x = NULL,
    y = "RMSE"
  )
gg

ggsave(
  filename = file.path("output", "figure", "NGA_RMSE_all_colored_by_adm1.png"),
  plot = gg, width = 6, height = 6, units = "in", dpi = 300, bg = "white"
)

# ============================================================================
# STEP 6: Compare GP with features vs GP without features
# ============================================================================

# Load GP predictions WITHOUT features
path_gp_no_features <- file.path("output", "gaussian_process", "NGA_gp_no_features_20251020_161159")

df_gp_no_features <- map_dfr(1:5, function(i) {
  path_i <- file.path(path_gp_no_features, "prior_posterior_diagnostics", sprintf("ppc_summary_cv%d.csv", i))

  read_csv(path_i, show_col_types = FALSE) %>%
    rename(
      y_hat   = y_pred_mean,
      ci_low  = `y_pred_ci_2.5`,
      ci_high = `y_pred_ci_97.5`
    ) %>%
    select(Datetime, y_hat, y_true, ci_low, ci_high, adm1_name) %>%
    mutate(
      cv_ind = i,
      model  = "gp_no_features"
    )
})

# Combine the two GP models with true values
df_gp_comparison <- bind_rows(
  df_gp |> mutate(model = "gp_with_features"),
  df_gp_no_features
) |>
  transmute(
    Datetime,
    adm1_name,
    cv_ind,
    model,
    cv_ind_adm1_name = paste0("CV ", cv_ind, ": ", adm1_name),
    poor_fcs = y_hat,
    ci_low,
    ci_high
  )

# Add true values
df_gp_all <- bind_rows(df_gp_comparison, df_true) %>%
  mutate(
    model = factor(
      model,
      levels = c("True", "gp_no_features","gp_with_features"),
      labels = c("True", "GP without covariates", "GP with covariates")
    )
  )

# Define colors for GP comparison
gp_comparison_cols <- c(
  "GP without covariates" = "#D95F02",
  "GP with covariates" = "#1B7837"
)

# Create comparison plot
gg_gp_comparison <- ggplot(df_gp_all, aes(
  x = Datetime, y = poor_fcs,
  color = model, linetype = model
)) +
  # Add confidence interval ribbons for GP models only
  geom_ribbon(
    data = df_gp_all %>% filter(model != "True") %>% 
      mutate(model = factor(model,
                            levels = c("GP without covariates", "GP with covariates"),
                            labels = c("GP without covariates", "GP with covariates")
                            )
             ),
    aes(ymin = ci_low, ymax = ci_high, fill = model),
    alpha = 0.3,
    color = NA
  ) +
  # Draw GP prediction lines
  geom_line(
    data = df_gp_all %>% filter(model != "True") %>%
      mutate(model = factor(model,
                            levels = c("GP without covariates", "GP with covariates"),
                            labels = c("GP without covariates", "GP with covariates")
      )
      ),
    linewidth = 0.6
  ) +
  # Draw true values last so they appear on top
  geom_line(
    data = df_gp_all %>% filter(model == "True"),
    linewidth = 0.7
  ) +
  # Create separate panel for each CV fold × region combination
  facet_wrap(~cv_ind_adm1_name, ncol = 4) +
  # Format y-axis as percentages
  scale_y_continuous(
    limits = c(0,1),
    labels = scales::percent_format(accuracy = 1)
  ) +
  # Set colors for each model
  scale_color_manual(
    values = c(
      "True" = "black",
      gp_comparison_cols
    )
  ) +
  scale_fill_manual(
    values = gp_comparison_cols
  ) +
  # Set line types (true values are dotted, predictions are solid)
  scale_linetype_manual(
    values = c(
      "True" = "dotted",
      "GP without covariates" = "solid",
      "GP with covariates" = "solid"
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

gg_gp_comparison

# Save the GP comparison plot
ggsave(
  filename = file.path("output", "figure", "NGA_GP_comparison_features_vs_no_features.png"),
  plot = gg_gp_comparison, width = 6, height = 6, units = "in", dpi = 300, bg = "white"
)

# ============================================================================
# STEP 7: Visualize GP with features normal centered (predictions only by region)
# ============================================================================

# Load GP predictions WITH features ridge centered
path_gp_ridge <- file.path("output", "gaussian_process", "NGA_gp_with_features_normal_centered_20260217_155435")
path_i <- file.path(path_gp_ridge, "prior_posterior_diagnostics", "ppc_summary_cv1.csv")

df_gp_ridge <- read_csv(path_i, show_col_types = FALSE) %>%
  rename(
    y_hat   = y_pred_mean,
    ci_low  = `y_pred_ci_2.5`,
    ci_high = `y_pred_ci_97.5`
  ) %>%
  select(Datetime, y_hat, y_true, ci_low, ci_high, adm1_name) %>%
  mutate(model = "gp_ridge_centered")

# Prepare data for visualization (predictions only, no true values)
df_gp_ridge_plot <- df_gp_ridge %>%
  transmute(
    Datetime,
    adm1_name,
    model,
    poor_fcs = y_hat,
    ci_low,
    ci_high
  ) %>%
  mutate(
    model = factor(
      model,
      levels = c("gp_ridge_centered"),
      labels = c("GP Ridge Centered")
    )
  )

# Define color for GP ridge centered model
gp_ridge_color <- "#1B7837"
lean_start1 <- as.Date("2022-05-01")
lean_end1   <- as.Date("2022-10-31")
lean_start2 <- as.Date("2023-05-01")
lean_end2   <- as.Date("2023-10-31")
# Create plot by region (predictions only)
gg_gp_ridge <- ggplot(df_gp_ridge_plot, aes(
  x = Datetime, y = poor_fcs)
  ) +
  geom_rect(
    inherit.aes = FALSE,
    xmin = lean_start1,
    xmax = lean_end1,
    ymin = -Inf,
    ymax = Inf,
    fill = "grey70",
    alpha = 0.01
  ) +
  geom_rect(
    inherit.aes = FALSE,
    xmin = lean_start2,
    xmax = lean_end2,
    ymin = -Inf,
    ymax = Inf,
    fill = "grey70",
    alpha = 0.01
  ) +
  # Add confidence interval ribbons
  geom_ribbon(
    aes(ymin = ci_low, ymax = ci_high),
    alpha = 0.3,
    fill = gp_ridge_color,
    color = NA
  ) +
  # Draw prediction lines
  geom_line(linewidth = 0.6, color =gp_ridge_color) +
  # Create separate panel for each region
  facet_wrap(~adm1_name, ncol = 5) +
  # Format y-axis as percentages
  scale_y_continuous(
    limits = c(0,1),
    labels = scales::percent_format(accuracy = 1)
  ) +
  theme_minimal(base_size = 10) +
  labs(
    x = NULL,
    y = "Prevalence of households \n with insufficient food consumption"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 6.5),
    axis.text.y = element_text(angle = 45, hjust = 1, size = 6.5),
    axis.title=element_text(size=7)
  ) 

gg_gp_ridge

# Save the GP ridge centered plot
ggsave(
  filename = file.path("output", "figure", "NGA_GP_ridge_centered_by_region.png"),
  plot = gg_gp_ridge, width = 6, height = 2.5, units = "in", dpi = 300, bg ="white"
)

# ============================================================================
# STEP 8: Map of nigeria
# ============================================================================
file_name <- file.path("data","new", "Nigeria" ,"Nigeria-weekly-with-features-region90.csv")
df_nigeria <- read_csv(file_name) |> 
  filter(data_type=="SURVEY 90 days") |>
  mutate(Datetime = as.Date(Datetime),
         poor_fcs = 100 * fcs,
         model = "True")  |>
  select(Datetime,adm1_name,model,poor_fcs)

df_spatial_map <- bind_rows(df_gp_ridge_plot |> mutate(poor_fcs = 100 * poor_fcs),
                            df_nigeria,
                            data.frame(
                              Datetime = rep(unique(df_true$Datetime), times=3),
                              adm1_name = rep(c("Adamawa","Yobe","Borno"),
                                              each = length(unique(df_true$Datetime))),
                              model = "Void"
                            )
                            ) |>
  mutate(
    type = factor(
      model,
      levels = c("GP Ridge Centered","True", "Void"),
      labels = c("Prediction","True", "Void")
    )
  ) 

nigeria_1 <- st_read("data/nigeria_geospatial") |> 
  select(state, geometry) |>
  mutate(adm1_name = str_to_title(state)) |>
  select(-state) 

nigeria_1$adm1_name[nigeria_1$adm1_name == "Fct"] <- "Abuja"
nigeria_1$adm1_name[nigeria_1$adm1_name == "Nasarawa"] <- "Nassarawa"
adm1_90days <- (df_spatial_map |> filter(type=="True") |> pull(adm1_name) |> unique()) 
adm1_30days <- c("Adamawa","Yobe","Borno")
adm1_unsurveyed <- (df_spatial_map |> filter(type=="Prediction") |> pull(adm1_name) |> unique())

df_spatial_map <- df_spatial_map |>
  left_join(nigeria_1, by = "adm1_name") |> 
  st_as_sf()

nigeria_1 <- nigeria_1 |> 
  mutate(survey_type = case_when(
    adm1_name %in% adm1_90days ~ "Covered",
    adm1_name %in% adm1_30days ~ "Covered high freq",
    .default = "Not covered"
  ))
nigeria_labels <- st_centroid(nigeria_1)
#### Plot with regions
nigeria_plot <- nigeria_1 |>
  mutate(
    covered_2 = if_else(survey_type == "Not covered", "Not covered", "Covered")
  )

p <- ggplot(nigeria_plot) +
  ggpattern::geom_sf_pattern(
    aes(fill = covered_2,
        alpha = survey_type,
        pattern = survey_type),         
    color = "gray30",
    size = 0.01,
    pattern_spacing = 0.01,             # controls the density of stripes
    pattern_angle   = 45,               # angle of the stripes
    pattern_fill    = "gray50",        # stripe colour (matches the Covered fill)
    pattern_colour  = "grey50"
  ) +
  # geom_sf(
  #   aes(fill = covered_2, alpha = survey_type),
  #   color = "gray30",
  #   size = 0.1
  # ) +
  geom_sf_text(
    data = nigeria_labels,
    aes(label = adm1_name),
    size = 2.5
  ) +
  scale_fill_manual(
    values = c(
      "Covered" = "#2C7FB8",
      "Not covered" = "grey70"
    ),
    name = "Survey status"
  ) +
  scale_alpha_manual(
    values = c(
      "Covered" = 1,
      "Covered high freq" = 0.6,  
      "Not covered" = 1
    ),
    guide = "none"  
  ) +
  scale_pattern_manual(
    values = c(
      "Not covered"        = "none",
      "Covered"            = "none",
      "Covered high freq"  = "stripe"  # pattern applied only to high freq
    ),
    guide = "none"                    # hide pattern legend, leaving only 2‑class fill
  ) +
  guides(
    fill = guide_legend(
      override.aes = list(
        pattern = "none",  # remove stripes in legend keys
        alpha   = 1        # make legend keys fully opaque
      )
    )
  ) +
  theme_minimal() +
  theme(
    legend.position = c(0.75, 0.15),  # inside bottom-right
    #axis.text = element_blank(),
    axis.title = element_blank()
  )
p
ggsave(
  filename = file.path("output", "figure", "NGA_map.png"),
  plot = p, width = 5.5, height = 4.5, units = "in", dpi = 300, bg ="white"
)

# spatial map oevr time
breaks_16w <- seq(from = min(df_nigeria$Datetime, na.rm = TRUE),
                 to   = max(df_nigeria$Datetime, na.rm = TRUE),
                 length.out = 12)
weeks <- df_spatial_map$Datetime |> unique() #63
breaks_weeks <- weeks[c(seq(1,63,by =6),63)]
p <- ggplot() +
  # Base layer (all states)
  geom_sf(
    data = df_spatial_map |> 
      filter(Datetime %in% breaks_weeks),
    aes(fill = poor_fcs),
    color = "gray30",
    linewidth = 0.1
  ) +
  
  # Overlay layer 
  geom_sf(
    data = df_spatial_map |> 
      filter(Datetime %in% breaks_weeks, type == "Prediction"),
    fill = NA,                 
    color = "black",
    linewidth =0.2               
  ) +
  facet_wrap(~Datetime, ncol = 4)  +
  scale_fill_gradientn(
    colours = wesanderson::wes_palette("Zissou1", n = 100, type = "continuous"),
    limits = c(0, 100),
    breaks = seq(0, 100, by = 20),
    labels = function(x) paste0(x, "%"),
    name = "Prevalence of households\n with insufficient food consumption  "
  ) +
  theme_void(base_size = 8) +
  guides(
    fill = guide_colorbar(
      direction = "vertical",
      title.position = "left",
      title.hjust = 0.5,
      title.theme = element_text(
        angle = 90,          # rotate anti-clockwise
        size = 7
      )
    )
  ) +
  theme(
    legend.position = "left",
    legend.text = element_text(angle=45, size = 5),
    legend.key.size = unit(0.4, 'cm'),
    legend.key.height= unit(0.8, 'cm'),
    legend.ticks.length = unit(c(-.05, 0), 'cm'), # different lengths for ticks on both sides
    legend.ticks = element_line(colour = "black"),
    axis.text = element_blank(),
  )
p
  
ggsave(
  filename = file.path("output", "figure", "NGA_spatial_plot.png"),
  plot = p, width = 5, height = 3, units = "in", dpi = 300, bg ="white"
)


### Regression coefficients
df_coef <- read_csv(file = file.path("output","gaussian_process",
                                     "NGA_gp_with_features_normal_centered_20260217_155435",
                                     "prior_posterior_diagnostics",
                                     "beta_summary_cv1.csv"))
feature_map <- c(
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

# 2) the exact order you want in the end
desired_order <- c(
  "Rainfall anomaly", "NDVI", "NDVI anomaly",
  "log (Currency exchange rate)",
  "log(Headline inflation)",
  "log(Food inflation rate)", "log(Food inflation rate) X MPI",
  "MPI", "log (Area size)", "Muslim population percentage",
  "log(Fatalities from explosion)", "log(Fatalities from battles)",
  "Number of conflicts (battle)",
  "Day index"
)

df_coef <- df_coef |>
  mutate(
    feature_clean = unname(feature_map[feature]),
    feature_clean = factor(feature_clean, levels = desired_order)
  ) |>
  arrange(feature_clean)

right_limit <- max(df_coef$ci_97.5) * 1.85

p <- df_coef |>
  mutate(
    feature_clean = fct_rev(fct_inorder(feature_clean)),
    # Create formatted label
    label_ci = sprintf(
      "%7.3f [%7.3f, %7.3f]",
      posterior_mean, ci_2.5, ci_97.5
    )
  ) %>%
  ggplot(aes(x = feature_clean, y = posterior_mean, fill = includes_zero)) +
  geom_col() +
  geom_errorbar(aes(ymin = ci_2.5, ymax = ci_97.5), 
                width = 0.2,alpha=0.5) +
  
  geom_text(
    aes(y = right_limit, label = label_ci),
    hjust = 1,
    size = 3.5
  )+
  
  geom_hline(yintercept = 0, linetype = "dashed") +
  coord_flip() +
  scale_y_continuous(limits = c(min(df_coef$ci_2.5), right_limit),
                     breaks = c(-0.5, 0.0, 0.5, 1.0)) +
  labs(x = NULL, y = "Posterior mean (95% CI)") +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text.y = element_text(hjust = 0)
  )
ggsave(
  filename = file.path("output", "figure", "NGA_coef.png"),
  plot = p, width = 7.5, height = 2.5, units = "in", dpi = 300, bg ="white"
)


