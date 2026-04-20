# Rscript TCD_gp_no_features_rho1fixed.R  > ../../../output/logs/TCD_gp_no_features_rho1fixed$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Load packages
library(here)

# Load common GP runner library
source(file.path(here(), "code", "models", "gaussian_process", "gp_runner_lib.R"))

# ==================== CONFIGURATION ====================
country <- "Chad"
iso3 <- "TCD"

# Cross-validation parameters
n_folds <- 1

# Stan sampling parameters
sampling_config <- list(
  rho1 = rep(350,6),
  Hurst2 = 0.5,
  warmup_init = 500,
  sampling_init = 1000,
  n_chains = 4,
  n_parallel_chains = 4,
  max_attempts = 3,
  y_transform = T
)

# No featuers selected
selected_features <- NULL

# Prior configuration
model_config <- list(
  prior_type = "no_prior"
)

# Stan file to use
stan_file_name <- "GPst_rho1fixed.stan"


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
