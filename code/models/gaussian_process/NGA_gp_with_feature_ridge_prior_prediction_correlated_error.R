# Rscript NGA_gp_with_feature_ridge_prior_prediction_correlated_error.R  > ../../../output/logs/NGA_gp_ridge_centered_prediction_correlated_error_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Load packages
library(here)

# Load common GP runner library
source(file.path(here(), "code", "models", "gaussian_process", "NGA_gp_runner_lib_prediction.R"))

# ==================== CONFIGURATION ====================
country <- "Nigeria"
iso3 <- "NGA"
type <- "PREDICTION"

# Cross-validation parameters
n_folds <- 5

# Stan sampling parameters
sampling_config <- list(
  rho1 = 300,
  Hurst2 = 0.5,
  warmup_init = 500,
  sampling_init = 1000,
  n_chains = 4,
  n_parallel_chains = 4,
  y_transform = TRUE,
  max_attempts = 4
)

# Prior configuration
model_config <- list(
  prior_type = "ridge_centered",
  hyper_prior_param = c("kappa"),
  country = "nigeria"
)

# Stan file to use: GP with covariates plus a region-specific overlap error
# term (tau_ov^2 * R_ov), accounting for dependence induced by overlapping
# 90-day rolling survey windows. Same mean and additive GP structure as
# GPstCOV_rho1fixed_ridge_centered_prior.stan.
stan_file_name <- "NGA_GPstCOV_rho1fixed_ridge_prior_sensitive_analysis.stan"

# Selected features (by SHAP), matching the calibrated correlated-error CV
# run (NGA_gp_with_feature_rho1fixed_ridge_prior_sensititvity_analysis.R)
selected_features <- c(
  "rainfall_3_months_anomaly_cubic_intp",
  "ndvi_value_cubic_intp",
  "ndvi_anomaly_cubic_intp",
  "log_currency_exchange",
  "log_Area",
  "MPI_intensity",
  "Muslim",
  "log_food_inflation_MPI",
  "log_fatalities_explosions",
  "MPI",
  "n_conflicts_Battles_rolsum_90days",
  "log_fatalities_battles",
  "headline_inflation_cubic_intp",
  "day_num"
)
# ==================== RUN MODEL ====================
results <- run_gp_model_overlap_prediction(
  country = country,
  iso3 = iso3,
  n_folds = n_folds,
  sampling_config = sampling_config,
  model_config = model_config,
  selected_features = selected_features,
  stan_file_name = stan_file_name
)
