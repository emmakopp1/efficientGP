# Rscript TCD_gp_with_features_rho1fixed_ridge_prior.R  > ../../../output/logs/TCD_gp_with_features_rho1fixed_ridge_prior_$(date +%Y%m%d_%H%M%S).log 2>&1 &


# Load packages
library(here)

# Load common GP runner library
source(file.path(here(), "code", "models", "gaussian_process", "gp_runner_lib.R"))

# ==================== CONFIGURATION ====================
country <- "Chad"
iso3 <- "TCD"

# Cross-validation parameters
n_folds <- 5

# Stan sampling parameters
sampling_config <- list(
  rho1 = rep(325, 6),
  Hurst2 = 0.5,
  warmup_init = 500,
  sampling_init = 1000,
  n_chains = 4,
  adapt_delta_init = 0.8,
  n_parallel_chains = 4,
  max_attempts = 1,
  y_transform = TRUE
)

# No featuers selected
# Features selected
selected_features <- c(
  "MPI_intensity",
  "currency_exchange_log",
  "ndvi_value_cubic_intp",
  "food_inflation_cubic_intp",
  "Waterways"
)

# Prior configuration
model_config <- list(
  country = iso3,
  prior_type = "ridge_centered",
  hyper_prior_param = "kappa"
)

# Stan file to use
stan_file_name <- "GPstCOV_rho1fixed_ridge_centered_prior.stan"


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
