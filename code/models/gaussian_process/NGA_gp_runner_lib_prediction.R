run_gp_model_prediction <- function(
  country,
  iso3,
  n_folds,
  sampling_config,
  model_config,
  selected_features,
  stan_file_name,
  topdir = here::here()
) {
  # Load required packages
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(bayesplot)
  library(cmdstanr)
  library(posterior)
  library(here)

  # Load utility functions for data processing, MCMC diagnostics, and visualization
  source(file.path(topdir, "code", "models", "utils", "convergence.R"))
  source(file.path(topdir, "code", "models", "utils", "data_prep.R"))
  source(file.path(topdir, "code", "models", "utils", "diagnostic.R"))

  # ========== DATA LOADING AND PREPROCESSING ==========
  # Load weekly timeseries data with covariate features
  file <- paste0(country, "-weekly-with-features.csv")
  df <- read.csv(
    file.path(topdir, "data", "new", country, file),
    header = TRUE
  )
  # Sort by region and datetime for spatio-temporal organization
  df <- df[order(df$adm1_name, df$Datetime), ]

  # Load static administrative region data (e.g., coordinates, region IDs)
  file <- paste0(country, "-static-data.csv")
  df_loc <- read.csv(
    file.path(topdir, "data", "new", country, file),
    header = TRUE
  )

  # ========== FEATURE ENGINEERING ==========
  # Apply feature transformations and create interactions:
  # - Ramadan × Muslim population percentage
  # - Food inflation × Multidimensional Poverty Index
  # - Log transforms for right-skewed variables
  # - Normalization of percentage features
  df <- create_features(df, country)
  # Validate that all requested features exist after preprocessing
  missing_features <- setdiff(selected_features, names(df))
  if (length(missing_features) > 0) {
    stop("Missing features after preprocessing: ", paste(missing_features, collapse = ", "))
  }

  # ========== CROSS-VALIDATION SETUP ==========
  # Filter to survey data (90-day observations) for CV structure
  # Change for SURVEY 90 days or PREDICTION
  # This ensures we have proper survey response annotations for each test fold
  region_90 <- df$adm1_code[df$data_type == "SURVEY 90 days"] |> unique()
  region_90_name <- df$adm1_name[df$data_type == "SURVEY 90 days"] |> unique()
  df_90 <- df[df$data_type == "SURVEY 90 days", ]

  # test set for nigeria
  region_pred <- df$adm1_code[df$data_type == "PREDICTION"] |> unique()
  region_pred_name <- df$adm1_name[df$data_type == "PREDICTION"] |> unique()
  df_pred <- df[df$data_type == "PREDICTION", ]


  # ========== STAN MODEL COMPILATION ==========
  # Load and compile the Stan probabilistic model
  # The Stan model defines the likelihood, priors, and parameters to estimate
  stan_path <- file.path(topdir, "code", "models", "stan")
  stan_file <- file.path(stan_path, stan_file_name)
  mod <- cmdstanr::cmdstan_model(
    stan_file,
    include_paths = stan_path
  )

  # Generate beta parameter names corresponding to selected features
  # Used for posterior extraction and diagnostic visualization
  betas <- paste0("beta", seq_along(selected_features))

  # ========== OUTPUT DIRECTORY STRUCTURE ==========
  # Create timestamped output directory to track model runs
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  output_path_top <- file.path(topdir, "output")

  # Determine model variant name for organization
  # "gp" for intercept-only models, "gp_with_features" when covariates are included
  model_variant <- if (length(selected_features) > 0) "gp_with_features" else "gp"

  output_path <- file.path(
    output_path_top,
    "gaussian_process",
    paste0(
      iso3, "_",
      model_variant, "_",
      model_config$prior_type, "_", timestamp, "_unsurveyed"
    )
  )
  if (!dir.exists(output_path)) dir.create(output_path, recursive = TRUE)

  # Convergence diagnostics: trace plots, Rhat/ESS plots, autocorrelation
  diagnostics_path <- file.path(output_path, "convergence_diagnostics")
  if (!dir.exists(diagnostics_path)) dir.create(diagnostics_path, recursive = TRUE)

  # Prior-posterior comparison and posterior predictive checks
  prior_posterior_path <- file.path(output_path, "prior_posterior_diagnostics")
  if (!dir.exists(prior_posterior_path)) dir.create(prior_posterior_path, recursive = TRUE)

  # Preprocessed train/test datasets for downstream deep learning applications
  path_data_train <- file.path(output_path, "data_train")
  path_data_test <- file.path(output_path, "data_test")

  # ========== INITIALIZE RESULT CONTAINERS ==========
  # Vectors and lists to accumulate results across CV folds
  rMSE <- c()
  MAE <- c()
  convergence_results <- list()
  beta_diagnostics_all <- list()
  ppc_diagnostics_all <- list()

  #y_transform <- sampling_config$y_transform

  # Save preprocessed train/test splits for external analysis
  #save_data_set(df_90, df_loc, i, selected_features, path_data_train, path_data_test)
    

  # ========== DATA STANDARDIZATION ========== 
  # Prepare training data: apply logit transform to target variable (FCS)
  # Standardize covariates to unit variance (zero mean, sd=1)
  # This prevents data leakage and improves MCMC sampling efficiency
  
  # Prepare test data using TRAINING set's standardization parameters
  data_list_tr <- prep_stan_data(
    df_90, df_loc, region_90,
    y_transform = sampling_config$y_transform, # Apply logit(FCS)
    features = selected_features,
    standardize = TRUE # Center and scale covariates
      )

    # Extract standardization parameters computed from TRAINING data
    # These will be applied to test data to maintain consistency
    cov_mean <- data_list_tr$cov_mean
    cov_sd <- data_list_tr$cov_sd
    
    #df_loc_prediction <- df_pred
    
    data_tes_process <- prep_stan_data_pred(
      df_pred, df_loc, region_pred,
      features = selected_features,
      standardize = TRUE,
      cov_mean = cov_mean,
      cov_sd = cov_sd
      )

    saveRDS(cov_mean, file = file.path(output_path, paste0("cov_mean", ".rds")))
    saveRDS(cov_sd, file = file.path(output_path, paste0("cov_sd", ".rds")))

    # Combine train and test data lists for Stan sampling
    data_list <- append(data_list_tr, data_tes_process[[1]])

    # Add spatial/temporal parameters to data list
    data_list_est <- append(
      data_list,
      list(
        rho1 = sampling_config$rho1, # Spatial range parameter
        Hurst2 = sampling_config$Hurst2 # Matérn smoothness
      )
    )

    n1 <- data_tes_process[[1]]$n1
    n2 <- data_tes_process[[1]]$n2

    # ========== GP DATA SIZE SUMMARY ==========
    # Display Gaussian Process data dimensions
    cat("  Gaussian Process Data Size:\n")
    cat("    - Spatial locations in the train set (N1):", data_list_tr$N1, "\n")
    cat("    - Spatial locations in the test set (n1):", data_tes_process[[1]]$n1, "\n")
    cat("    - Temporal observations per location in the train (N2):", data_list$N2, "\n")
    cat("    - Temporal observations per location in the test (n2):", data_tes_process[[1]]$n2, "\n")
    cat("    - Total observations in the train:", data_list_tr$N1 * data_list$N2, "\n")
    cat("    - Total observations in the test:", data_tes_process[[1]]$n1 * data_tes_process[[1]]$n2, "\n")
    cat("    - Value of rho1 parameter:", sampling_config$rho1, "\n")
    if (length(selected_features) > 0) {
      cat("    - Number of features:", length(selected_features), "\n")
    }
    cat("\\n")

    # ========== STAN SAMPLING WITH ADAPTIVE CONVERGENCE ==========
    # Fit the model using adaptive MCMC sampling
    # Automatically adjusts sampler parameters if convergence criteria not met
    fit <- adaptive_sample(
      mod,
      data_list_est,
      seed = 1, # Fold-specific seed for reproducibility
      max_attempts = sampling_config$max_attempts,
      warmup_init = sampling_config$warmup_init,
      sampling_init = sampling_config$sampling_init,
      n_chains = sampling_config$n_chains,
      n_parallel_chains = sampling_config$n_parallel_chains,
      adapt_delta_init = sampling_config$adapt_delta_init
    )

    fit$save_object(file = file.path(output_path, paste0("fit.rds")))
    # ========== DIAGNOSTIC GENERATION ==========
    i <-1 
    # Check convergence criteria: Rhat < 1.05, ESS_bulk > 400, no divergences
    convergence_results[[as.character(i + 1)]] <- check_convergence(fit, i, diagnostics_path)

    # Generate visualization plots: trace plots, Rhat/ESS distributions, ACF
    create_convergence_plots(
      fit, i, diagnostics_path,
      selected_features = selected_features,
      hyperprior = model_config$hyper_prior_param
    )

    # ========== POSTERIOR ANALYSIS ==========
    # Extract feature coefficient diagnostics (beta parameters)
    # Computes posterior mean, sd, credible intervals, Rhat, ESS for each feature
    beta_diagnostics_all[[as.character(i + 1)]] <- extract_beta_diagnostics(
      fit = fit,
      cv_fold = i,
      prior_posterior_path = prior_posterior_path,
      selected_features = selected_features
    )

    # Extract posterior mean predictions for all parameters
    # Used for prediction analysis and result visualization
    extract_posterior_mean(
      fit = fit,
      cv_fold = i,
      output_path = output_path
    )

    # Extract Log-Likelihood posterior mean predictions for all parameters
    # Used for prediction analysis and result visualization
    post_mean_llkd <- extract_posterior_mean_llkd(
      fit = fit,
      cv_fold = i,
      output_path = output_path
    )

    # ========== MODEL EVALUATION ==========
    # Extract posterior predictive samples and compute prediction intervals
    # Calculate residuals: y_test - predicted values
    # Assess 95% CI coverage and prediction accuracy
    ppc_diagnostics_all[[as.character(i + 1)]] <- extract_posterior_predictive_data(
      fit = fit,
      y_true = data_tes_process[[2]],
      cv_fold = i,
      prior_posterior_path = prior_posterior_path,
      n1 = n1,
      n2 = n2,
      y_transform = sampling_config$y_transform ,
      adm1_code = rep(region_pred, each = n2),
      adm1_name = rep(region_pred_name, each = n2),
      datetime = rep(as.Date(df$Datetime[1:n2], tz = "GMT"), times = n1)
    )

    # Calculate performance metrics for this fold
    rMSE <- c(rMSE, sqrt(mean(ppc_diagnostics_all[[as.character(i + 1)]]$residual^2)))
    MAE <- c(MAE, mean(abs(ppc_diagnostics_all[[as.character(i + 1)]]$residual)))
    

}
