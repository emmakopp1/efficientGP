# Rscript NGA_gp_no_features_rho1fixed.R  > ../../../output/logs/NGA_gp_no_features_rho1fixed_$(date +%Y%m%d_%H%M%S).log 2>&1 &

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
  max_attempts = 3,
  y_transform = TRUE
)

# No featuers selected
selected_features <- NULL

# Prior configuration
model_config <- list(
  prior_type = "no_feature"
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
