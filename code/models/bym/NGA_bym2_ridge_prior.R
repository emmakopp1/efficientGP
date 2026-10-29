# Rscript code/models/bym/NGA_bym2_ridge_prior.R > output/logs/NGA_bym2_ridge_prior_$(date +%Y%m%d_%H%M%S).log 2>&1 &

library(here)
source(file.path(here(), "code", "models", "bym", "bym_runner_lib.R"))

# ==================== CONFIGURATION ====================
country <- "Nigeria"
iso3 <- "NGA"

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
# (NGA_gp_with_feature_rho1fixed_normal_prior.R)
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
results <- run_bym_model(
  country = country,
  iso3 = iso3,
  n_folds = n_folds,
  sampling_config = sampling_config,
  model_config = model_config,
  selected_features = selected_features,
  stan_file_name = "BYM2_ridge_prior.stan"
)
