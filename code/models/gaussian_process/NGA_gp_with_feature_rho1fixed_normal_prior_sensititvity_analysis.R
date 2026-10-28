# Rscript NGA_gp_with_feature_rho1fixed_normal_prior_sensititvity_analysis.R > ../../../output/logs/NGA_gp_with_feature_rho1fixed_normal_prior_sensititvity_analysis_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Load packages
library(here)

# Load common GP runner library
source(file.path(here(), "code", "models", "gaussian_process", "gp_runner_lib.R"))

# ==================== CONFIGURATION ====================
# Sensitivity analysis: dependence induced by overlapping 90-day rolling
# survey windows, modelled with a region-specific overlap error
# tau_ov^2 * R_ov (see GPstCOV_rho1fixed_normal_prior_overlap.stan).
# Same data, folds, features and sampling settings as the reference model
# NGA_gp_with_feature_rho1fixed_normal_prior.R, for comparability.
country <- "Nigeria"
iso3 <- "NGA"

# Cross-validation parameters
n_folds <- 5

# Stan sampling parameters
sampling_config <- list(
  rho1 = rep(300, 6),
  Hurst2 = 0.5,
  warmup_init = 500,
  sampling_init = 1000,
  n_chains = 4,
  n_parallel_chains = 4,
  y_transform = TRUE,
  max_attempts = 3,
  y_transform = TRUE
)

# Prior configuration
model_config <- list(
  prior_type = "normal_centered",
  hyper_prior_param = "beta_sigma",
  country = "nigeria"
)

# Stan file to use
stan_file_name <- "GPstCOV_rho1fixed_normal_prior_overlap.stan"

# Selected features (by SHAP)
# Matches the reference run NGA_gp_with_features_normal_centered_20260217_155435
selected_features <- c(
  "rainfall_3_months_anomaly_cubic_intp",
  "ndvi_value_cubic_intp",
  "ndvi_anomaly_cubic_intp",
  "log_currency_exchange",
  "log_Area",
  #"Muslim",
  "log_food_inflation",
  "log_food_inflation_MPI",
  "log_fatalities_explosions",
  "MPI",
  "n_conflicts_Battles_rolsum_90days",
  "log_fatalities_battles",
  "headline_inflation_cubic_intp",
  "day_num"
)

# ==================== RUN MODEL ====================
results <- run_gp_model_overlap(
  country = country,
  iso3 = iso3,
  n_folds = n_folds,
  sampling_config = sampling_config,
  model_config = model_config,
  selected_features = selected_features,
  stan_file_name = stan_file_name
)
