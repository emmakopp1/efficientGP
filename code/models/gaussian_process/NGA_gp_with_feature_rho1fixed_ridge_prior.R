# Rscript NGA_gp_with_feature_rho1fixed_ridge_prior.R > ../../../output/logs/NGA_gp_with_feature_rho1fixed_ridge_prior_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Load packages
library(here)

# Load common GP runner library
source(file.path(here(), "code", "models", "gaussian_process", "gp_runner_lib.R"))

# ==================== CONFIGURATION ====================
country <- "Nigeria"
iso3 <- "NGA"

# Cross-validation parameters
n_folds <- 1

# Stan sampling parameters
sampling_config <- list(
  rho1 = rep(300, 6),
  Hurst2 = 0.5,
  warmup_init = 100,
  sampling_init = 400,
  n_chains = 4,
  n_parallel_chains = 4,
  y_transform = TRUE,
  max_attempts = 3
)

# Prior configuration
model_config <- list(
  prior_type = "ridge_centered",
  hyper_prior_param = c("kappa"),
  country = "nigeria"
)

# Stan file to use
stan_file_name <- "GPstCOV_rho1fixed_ridge_centered_prior.stan"

# Selected features (by SHAP)
selected_features <- c(
  "rainfall_3_months_anomaly_cubic_intp",
  "ndvi_value_cubic_intp",
  "ndvi_anomaly_cubic_intp",
  "log_currency_exchange",
  "log_Area",
  "Muslim",
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
results <- run_gp_model(
  country = country,
  iso3 = iso3,
  n_folds = n_folds,
  sampling_config = sampling_config,
  model_config = model_config,
  selected_features = selected_features,
  stan_file_name = stan_file_name
)
