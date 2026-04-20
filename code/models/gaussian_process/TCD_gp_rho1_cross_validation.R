# nohup Rscript TCD_gp_rho1_cross_validation.R > ../../../output/logs/TCD_gp_rho1_cross_validation$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Load packages
library(here)
source(file.path(here(), "code", "models", "gaussian_process", "gp_runner_nested_cross_validation.R"))

# Load common GP runner library
rho1_grid <- c(1300/4, 1300/2, 1300, 1300*2) 

# ==================== CONFIGURATION ====================
country <- "Chad"
iso3 <- "TCD"

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
  hyper_prior_param = "beta_sigma"
)

# Stan file to use
stan_file_name <- "GPstCOV_rho1fixed_ridge_centered_prior.stan"

# Selected features (by SHAP)
selected_features <- c(
  "MPI_intensity",
  "currency_exchange_log",
  "ndvi_value_cubic_intp",
  "food_inflation_cubic_intp",
  "Waterways"
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
