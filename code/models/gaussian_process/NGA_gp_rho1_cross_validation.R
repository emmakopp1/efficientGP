# nohup Rscript NGA_gp_rho1_cross_validation.R > ../../../output/logs/NGA_gp_rho1_cross_validation_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Load packages
library(here)
source(file.path(here(), "code", "models", "gaussian_process", "gp_runner_nested_cross_validation.R"))

# Load common GP runner library
rho1_grid <- c(600 / 2, 600, 2 * 600) 

# ==================== CONFIGURATION ====================
country <- "Nigeria"
iso3 <- "NGA"

# Cross-validation parameters
n_folds <- 1

# Fixer rho1
sampling_config <- list(
  rho1 = rho1_grid,
  Hurst2 = 0.5,
  warmup_init = 100,
  sampling_init = 500,
  n_chains = 4,
  n_parallel_chains = 4,
  max_attempts = 3,
  y_transform = TRUE
)

model_config <- list(
  prior_type = "normal",
  hyper_prior_param = "beta_sigma",
  country = "nigeria"
)

# Stan file to use
stan_file_name <- "GPstCOV_rho1fixed_normal_prior.stan"

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
results <- run_gp_model_cv(
  country = country,
  iso3 = iso3,
  n_folds = n_folds,
  sampling_config = sampling_config,
  model_config = model_config,
  selected_features = selected_features,
  stan_file_name = stan_file_name
)
