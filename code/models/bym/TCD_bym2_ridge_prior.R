# Rscript code/models/bym/TCD_bym2_ridge_prior.R > output/logs/TCD_bym2_ridge_prior_$(date +%Y%m%d_%H%M%S).log 2>&1 &

library(here)
source(file.path(here(), "code", "models", "bym", "bym_runner_lib.R"))

# ==================== CONFIGURATION ====================
country <- "Chad"
iso3 <- "TCD"

# Same folds as the GP (cv_ind 0 to 5)
n_folds <- 5

sampling_config <- list(
  # The temporal RW1 is partly confounded with the national (time-only)
  # covariates, which slows mixing of nu and of their coefficients:
  # longer chains than the GP are needed to reach ESS > 400
  warmup_init = 1000,
  sampling_init = 4500,
  n_chains = 4,
  n_parallel_chains = 4,
  adapt_delta_init = 0.9,
  max_attempts = 2,
  y_transform = TRUE # model logit(FCS), as the GP
)

model_config <- list(
  prior_type = "ridge",
  pc_u = 1, # PC priors: P(sd > 1) = 0.01
  pc_alpha = 0.01
)

# Same features as the GP reported in the paper
# (TCD_gp_with_features_rho1fixed_ridge_prior.R)
selected_features <- c(
  "MPI_intensity",
  "currency_exchange_log",
  "ndvi_value_cubic_intp",
  "food_inflation_cubic_intp",
  "Waterways"
)

# ==================== RUN MODEL ====================
results <- run_bym_model(
  country = country,
  iso3 = iso3,
  n_folds = n_folds,
  sampling_config = sampling_config,
  model_config = model_config,
  selected_features = selected_features,
  stan_file_name = "BYM2_ridge_prior.stan"
)
