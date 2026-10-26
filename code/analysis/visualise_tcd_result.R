# ============================================================================
# Chad (TCD) Food Insecurity Prediction Model Comparison
# ============================================================================
# This script compares 3 different models for predicting food insecurity in Chad:
# 1. Gaussian Process (GP)
# 2. Bayesian Ridge Regression
# 3. Multi-Layer Perceptron (MLP)
#
# It generates:
# - Time series plots of predictions vs. true values
# - Performance metrics (RMSE, MAE, coverage, interval scores)
# - Comparison plots by region and cross-validation fold (6 folds, 0-5)
# ============================================================================

library(tidyverse)
library(scales)
library(purrr)
library(stringr)
library(here)
library(sf)


# Define paths to model output directories
path_tcd <- c(
  file.path(
    "output", "gaussian_process",
    "TCD_gp_with_features_ridge_centered_20251216_114735"
  ),
  file.path("output", "data", "tcd_bayesian_ridge"),
  file.path("output", "data", "tcd_mlp"),
  file.path("output","data","tcd_xg_boost")
)

# ============================================================================
# STEP 1: Load predictions from all models (6-fold cross-validation: 0-5)
# ============================================================================

# ---- Bayesian Ridge ----
# Load Bayesian Ridge predictions for each CV fold (0 to 5)
path <- path_tcd[[2]]
df_bayesian_ridge <- map_dfr(0:5, function(i) {
  read_csv(file.path(path, sprintf("tcd_predictions_fold%d.csv", i)), show_col_types = FALSE) %>%
    mutate(
      cv_ind = i,
      model  = "bayesian ridge"
    )
})

# ---- MLP ----
# Load MLP predictions and select relevant columns
path <- path_tcd[[3]]

df_mlp <- map_dfr(0:5, function(i) {
  read_csv(file.path(path, sprintf("predictions_cv%d.csv", i)), show_col_types = FALSE) %>%
    select(Datetime, y_hat, y_true, ci_low, ci_high, adm1_name) %>%
    mutate(
      cv_ind = i,
      model  = "mlp"
    )
})

# ---- Gaussian process ----
# Load GP predictions and rename columns to match other models
path <- path_tcd[[1]]

df_gp <- map_dfr(0:5, function(i) {
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
path <- path_tcd[[4]]

df_xg_boost <- map_dfr(0:5, function(i) {
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
        "GP",
        "Bayesian Ridge",
        "MLP",
        "XGBoost"
      )
    )
  )

# ============================================================================
# STEP 3: Create aligned facet grid (4 columns per row)
# ============================================================================
# Chad has fewer regions than Nigeria, so we need to pad empty panels
# to maintain a consistent 4-column layout across all CV folds

# Build facet labels
df_all <- df_all |>
  mutate(
    cv_ind = as.integer(cv_ind),
    cv_ind_adm1_name = paste0("CV ", cv_ind, ": ", adm1_name)
  )

# Create full set of facet levels: each CV fold gets exactly 4 panel slots
# If a CV fold has < 4 regions, add padding panels
panel_levels <- df_all |>
  distinct(cv_ind, adm1_name, cv_ind_adm1_name) |>
  arrange(cv_ind, adm1_name) |>
  group_by(cv_ind) |>
  summarise(
    lev = {
      x <- cv_ind_adm1_name
      k <- length(x)
      if (k < 4) {
        # Add padding panels to complete the row
        pads <- paste0("CV ", unique(cv_ind), ": __pad", seq_len(4 - k))
        c(x, pads)
      } else {
        x
      }
    },
    .groups = "drop"
  ) |>
  pull(lev) |>
  unlist()

# Apply factor levels to control facet panel ordering
df_all <- df_all |>
  mutate(cv_ind_adm1_name = factor(cv_ind_adm1_name, levels = panel_levels))

# Custom labeller: hide labels for padding panels
lab_hide_pad <- function(x) ifelse(str_detect(x, "__pad"), "", x)


model_cols <- c(
  "GP" = "#1B7837",
  "Bayesian Ridge" = "#D73027",
  "MLP" = "#2C7FB8",
  "XGBoost" = "#8E63CE"
)


gg <- ggplot(df_all, aes(x = Datetime, y = poor_fcs, color = model, linetype = model)) +
  geom_ribbon(
    data = df_all |> filter(model != "True"),
    aes(ymin = ci_low, ymax = ci_high, fill = model),
    alpha = 0.3,
    color = NA
  ) +
  geom_line(data = df_all |> filter(model != "True"), linewidth = 0.6) +
  geom_line(data = df_all |> filter(model == "True"), linewidth = 0.7) +
  facet_wrap(
    ~cv_ind_adm1_name,
    ncol = 4,
    drop = FALSE,
    labeller = as_labeller(lab_hide_pad)
  ) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
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
      "GP" = "solid",
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

ggsave(
  filename = file.path("output", "figure", "TCD_fcs_prev_all_by_adm1.png"),
  plot = gg, width = 6, height = 7, units = "in", dpi = 300, bg = "white"
)

# ============================================================================
# STEP 5: Calculate performance metrics for each model
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
        "GP",
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
  ) |>
  mutate(cv_ind_adm1_name = factor(cv_ind_adm1_name, levels = panel_levels))


gg <- ggplot(df_plot_metrics |> filter(metric == "RMSE"), aes(x = model, y = value)) +
  geom_col(aes(fill = is_best), width = 0.8, color = "white") +
  facet_wrap(
    ~cv_ind_adm1_name,
    ncol = 4,
    drop = FALSE,
    labeller = as_labeller(lab_hide_pad)
  ) +
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
  filename = file.path("output", "figure", "TCD_RMSE_all_by_adm1.png"),
  plot = gg, width = 6, height = 7.5, units = "in", dpi = 300, bg = "white"
)

df_plot_rmse <- df_all_metrics |>
  mutate(cv_ind_adm1_name = paste0("CV ", cv_ind, ": ", adm1_name)) |>
  group_by(cv_ind, adm1_name) |>
  mutate(is_best = rmse == min(rmse, na.rm = TRUE)) |>
  ungroup() |>
  mutate(cv_ind_adm1_name = factor(cv_ind_adm1_name, levels = panel_levels))

gg <- ggplot(
  df_plot_rmse,
  aes(x = model, y = rmse, fill = model, alpha = is_best)
) +
  geom_col(width = 0.8, color = "white") +
  facet_wrap(
    ~cv_ind_adm1_name,
    ncol = 4,
    drop = FALSE,
    labeller = as_labeller(lab_hide_pad)
  ) +
  scale_fill_manual(values = model_cols) +
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
  filename = file.path("output", "figure", "TCD_RMSE_all_colored_by_adm1.png"),
  plot = gg, width = 6, height = 7, units = "in", dpi = 300, bg = "white"
)

# ============================================================================
# STEP 7: Compare GP without features vs GP with features
# ============================================================================

# Load GP predictions WITHOUT features
path_gp_no_features <- file.path("output", "gaussian_process", "TCD_gp_no_prior_20251210_112502")

df_gp_no_features <- map_dfr(0:5, function(i) {
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

# Combine the two GP models
df_gp_comparison <- bind_rows(
  df_gp %>% mutate(model = "gp_with_features"),
  df_gp_no_features
) %>%
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
      levels = c("True", "gp_with_features", "gp_no_features"),
      labels = c("True", "GP with features", "GP without features")
    )
  )

# Apply same panel levels for consistent layout with padding
df_gp_all <- df_gp_all %>%
  mutate(
    cv_ind = as.integer(cv_ind),
    cv_ind_adm1_name = factor(cv_ind_adm1_name, levels = panel_levels)
  )

# Define colors for GP comparison (same as Nigeria: orange for with, green for without)
gp_comparison_cols <- c(
  "GP with features" = "#D95F02",
  "GP without features" = "#1B7837"
)

# Create comparison plot
gg_gp_comparison <- ggplot(df_gp_all, aes(
  x = Datetime, y = poor_fcs,
  color = model, linetype = model
)) +
  # Add confidence interval ribbons for GP models only
  geom_ribbon(
    data = df_gp_all %>% filter(model != "True"),
    aes(ymin = ci_low, ymax = ci_high, fill = model),
    alpha = 0.3,
    color = NA
  ) +
  # Draw GP prediction lines
  geom_line(
    data = df_gp_all %>% filter(model != "True"),
    linewidth = 0.6
  ) +
  # Draw true values last so they appear on top
  geom_line(
    data = df_gp_all %>% filter(model == "True"),
    linewidth = 0.7
  ) +
  # Create facet grid with padding panels (4 columns per row)
  facet_wrap(
    ~cv_ind_adm1_name,
    ncol = 4,
    drop = FALSE,  # Keep empty panels for alignment
    labeller = as_labeller(lab_hide_pad)  # Hide padding labels
  ) +
  # Format y-axis as percentages
  scale_y_continuous(
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
      "GP with features" = "solid",
      "GP without features" = "solid"
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
  filename = file.path("output", "figure", "TCD_GP_comparison_features_vs_no_features.png"),
  plot = gg_gp_comparison, width = 6, height = 7, units = "in", dpi = 300, bg = "white"
)

# ============================================================================
# STEP 8: Chad CV fold re-assignment - region selection for fold 0
# ============================================================================
# Fold 0 ("fixed") always stays in the train set and is never used as a test
# fold. It contains N'Djamena (capital) and any region whose geographic
# quadrant (North/South/East/West) has more than 5 regions, so that folds
# 1-5 can each contain exactly one region per quadrant (mirroring the
# Nigeria nc/ne/nw/s x cv_ind design, see STEP 8 in visualise_nga_result.R).
#
# STEP 8a: how many regions are needed in fold 0
# STEP 8b: classify regions into geographic quadrants (N / S / E / W)
# STEP 8c: spatial map of FCS prevalence over time, used to pick the
#          region(s) to add to fold 0 besides the capital
# ============================================================================

df_static_tcd <- read_csv(file.path("data", "new", "chad", "chad-static-data.csv"), show_col_types = FALSE)
df_weekly_tcd <- read_csv(file.path("data", "new", "chad", "chad-weekly-with-features.csv"), show_col_types = FALSE)

capital_tcd <- "Ndjamzna" # N'Djamena, as spelled in the Chad CSVs

# ---- STEP 8a: fold 0 size ----
# Tibesti is already excluded upstream (not present in chad-weekly-with-features.csv)
n_total_tcd <- df_static_tcd |>
  filter(adm1_name != "Tibesti") |>
  pull(adm1_name) |>
  n_distinct()
n_folds_1to5 <- 5
n_quadrants <- 4
n_in_folds_1to5 <- n_folds_1to5 * n_quadrants # one region per quadrant per fold
n_fold0 <- n_total_tcd - n_in_folds_1to5

cat("STEP 8a: Fold 0 size\n")
cat("  Total regions available (excl. Tibesti):", n_total_tcd, "\n")
cat("  Regions needed for folds 1-5 (", n_folds_1to5, "folds x", n_quadrants, "quadrants):", n_in_folds_1to5, "\n")
cat("  => Regions to place in fold 0 (incl. capital):", n_fold0, "\n\n")

# ---- STEP 8b: quadrant classification ----
# Quadrant assigned relative to the median lat/lon of all non-capital,
# non-Tibesti regions (N'Djamena and Tibesti excluded from the reference
# center so the capital doesn't skew the split).
df_geo_tcd <- df_static_tcd |>
  filter(!(adm1_name %in% c("Tibesti", capital_tcd)))

lat_center_tcd <- median(df_geo_tcd$Lat)
lon_center_tcd <- median(df_geo_tcd$Lon)

df_geo_tcd <- df_geo_tcd |>
  mutate(
    dlat = Lat - lat_center_tcd,
    dlon = Lon - lon_center_tcd,
    quadrant = case_when(
      abs(dlat) > abs(dlon) & dlat > 0 ~ "North",
      abs(dlat) > abs(dlon) & dlat <= 0 ~ "South",
      dlon > 0 ~ "East",
      TRUE ~ "West"
    )
  )

cat("STEP 8b: Quadrant classification (excl. capital & Tibesti)\n")
print(df_geo_tcd |> select(adm1_name, Lat, Lon, quadrant) |> arrange(quadrant, adm1_name), n = Inf)
cat("\n")
print(count(df_geo_tcd, quadrant))
cat("\n  Need exactly", n_folds_1to5, "regions per quadrant for folds 1-5;\n")
cat("  any quadrant with more than", n_folds_1to5, "regions has surplus candidate(s) for fold 0.\n\n")

# ---- STEP 8c: spatial map of FCS prevalence over time ----
# Mirrors the Nigeria spatial map (STEP 8 in visualise_nga_result.R /
# output/figure/NGA_spatial_plot.png), here used to visually assess which
# region(s), besides the capital, look the most unstable over time and are
# therefore good candidates for fold 0.
shp_name_fix_tcd <- c(
  "Barh El Gazel" = "Barh El Gazal",
  "Guéra" = "Guera",
  "N'djamena" = capital_tcd,
  "Ouaddaï" = "Ouaddai",
  "Tandjilé" = "Tandjile"
)

gdf_tcd <- st_read(file.path("data", "chad_geospatial", "tcd_admin1.shp"), quiet = TRUE) |>
  mutate(adm1_name = str_to_title(adm1_name)) |>
  mutate(adm1_name = recode(adm1_name, !!!shp_name_fix_tcd)) |>
  filter(adm1_name != "Tibesti") |>
  select(adm1_name, geometry)

df_weekly_tcd <- df_weekly_tcd |>
  mutate(
    Datetime = as.Date(Datetime),
    poor_fcs = 100 * fcs
  )

# Pick 12 evenly spaced weeks across the period, like the Nigeria plot
weeks_tcd <- sort(unique(df_weekly_tcd$Datetime))
step_tcd <- max(1, floor(length(weeks_tcd) / 12))
breaks_weeks_tcd <- weeks_tcd[seq(1, length(weeks_tcd), by = step_tcd)][1:12]

df_map_tcd <- df_weekly_tcd |>
  filter(Datetime %in% breaks_weeks_tcd) |>
  inner_join(gdf_tcd, by = "adm1_name") |>
  st_as_sf()

capital_outline_tcd <- df_map_tcd |> filter(adm1_name == capital_tcd)

p_tcd_spatial <- ggplot() +
  geom_sf(
    data = df_map_tcd,
    aes(fill = poor_fcs),
    color = "gray30",
    linewidth = 0.1
  ) +
  geom_sf(
    data = capital_outline_tcd,
    fill = NA,
    color = "black",
    linewidth = 0.5
  ) +
  facet_wrap(~Datetime, ncol = 4) +
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
        angle = 90,
        size = 7
      )
    )
  ) +
  theme(
    legend.position = "left",
    legend.text = element_text(angle = 45, size = 5),
    legend.key.size = unit(0.4, "cm"),
    legend.key.height = unit(0.8, "cm"),
    legend.ticks.length = unit(c(-.05, 0), "cm"),
    legend.ticks = element_line(colour = "black"),
    axis.text = element_blank(),
  )
p_tcd_spatial

ggsave(
  filename = file.path("output", "figure", "TCD_spatial_plot.png"),
  plot = p_tcd_spatial, width = 5, height = 3, units = "in", dpi = 300, bg = "white"
)

# ---- STEP 8d: FCS prevalence summary per region (sorted by variability) ----
# Complements the map: mean/sd/min/max FCS prevalence over time per region,
# used alongside the map to assess which region(s) look most unstable.
summary_tcd <- df_weekly_tcd |>
  group_by(adm1_name) |>
  summarise(
    mean = round(mean(poor_fcs), 1),
    std = round(sd(poor_fcs), 1),
    min = round(min(poor_fcs), 1),
    max = round(max(poor_fcs), 1),
    .groups = "drop"
  ) |>
  left_join(df_geo_tcd |> select(adm1_name, quadrant), by = "adm1_name") |>
  arrange(desc(std))

cat("STEP 8d: FCS prevalence summary per region (sorted by variability)\n")
print(summary_tcd, n = Inf)
