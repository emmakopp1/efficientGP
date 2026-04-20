run_gp_model <- function(
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
  mod <- cmdstanr::cmdstan_model(stan_file,include_paths =  stan_path)

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
      model_config$prior_type, "_", timestamp
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

  # ========== CROSS-VALIDATION LOOP ==========
  # Implements region-based leave-one-region-out cross-validation
  # For each fold, one region is held out for testing, others used for training
  for (i in 0:1) {
    cat("\n\n=============================================\n")
    cat("         CROSS-VALIDATION FOLD", i, "\n")
    cat("=============================================\n\n")

    # ========== DATA SPLIT ==========
    # Identify test region for this fold

    # Training data: all other regions
    if (country == "Nigeria") {
      # train region
      train_region <- df_90$adm1_code[df_90$cv_ind != i] |> unique()
      train_region_name <- df_90$adm1_name[df_90$cv_ind != i] |> unique()

      # test region
      test_region <- df_90$adm1_code[df_90$cv_ind == i] |> unique()
      test_region_name <- df_90$adm1_name[df_90$cv_ind == i] |> unique()
      if (length(test_region_name) == 0) next

      # pred region
      pred_region <- df_pred |>
        pull(adm1_code) |>
        unique()
      pred_region_name <- df_pred |>
        pull(adm1_name) |>
        unique()

      y_transform <- sampling_config$y_transform

      # Save preprocessed train/test splits for external analysis
      save_data_set(df_90, df_loc, i, selected_features, path_data_train, path_data_test)
    } else if (country == "Chad") {
      # train region
      train_region <- df$adm1_code[df$cv_ind != (i)] |> unique()
      train_region_name <- df$adm1_name[df$cv_ind != (i)] |> unique()

      # test region
      test_region <- df$adm1_code[df$cv_ind == (i)] |> unique()
      test_region_name <- df$adm1_name[df$cv_ind == (i)] |> unique()
      if (length(test_region_name) == 0) next

      y_transform <- sampling_config$y_transform

      # Save preprocessed train/test splits for external analysis
      save_data_set(df, df_loc, i, selected_features, path_data_train, path_data_test)
    }


    # ========== DATA STANDARDIZATION ==========
    # Prepare training data: apply logit transform to target variable (FCS)
    # Standardize covariates to unit variance (zero mean, sd=1)
    # This prevents data leakage and improves MCMC sampling efficiency

    # Prepare test data using TRAINING set's standardization parameters
    # Prevents test data information leakage into model fitting
    if (country == "Chad") {
      data_list_tr <- prep_stan_data_names(
        df_long = df,
        df_loc = df_loc,
        regions_name = train_region_name,
        y_transform = sampling_config$y_transform, # Apply logit(FCS)
        features = selected_features,
        standardize = TRUE # Center and scale covariates
      )

      # Extract standardization parameters computed from TRAINING data
      # These will be applied to test data to maintain consistency
      cov_mean <- data_list_tr$cov_mean
      cov_sd <- data_list_tr$cov_sd

      data_tes_process <- prep_stan_data_pred_name(
        df, df_loc, test_region_name,
        features = selected_features,
        standardize = TRUE,
        cov_mean = cov_mean,
        cov_sd = cov_sd
      )
    } else if (country == "Nigeria") {
      data_list_tr <- prep_stan_data(
        df_90, df_loc, train_region,
        y_transform = sampling_config$y_transform, # Apply logit(FCS)
        features = selected_features,
        standardize = TRUE # Center and scale covariates
      )

      # Extract standardization parameters computed from TRAINING data
      # These will be applied to test data to maintain consistency
      cov_mean <- data_list_tr$cov_mean
      cov_sd <- data_list_tr$cov_sd
      data_tes_process <- prep_stan_data_pred(
        df_90, df_loc, test_region,
        features = selected_features,
        standardize = TRUE,
        cov_mean = cov_mean,
        cov_sd = cov_sd
      )
    }

    saveRDS(cov_mean, file = file.path(output_path, paste0("cov_mean_fold_", i, ".rds")))
    saveRDS(cov_sd, file = file.path(output_path, paste0("cov_sd_fold_", i, ".rds")))

    # Combine train and test data lists for Stan sampling
    data_list <- append(data_list_tr, data_tes_process[[1]])

    # Add spatial/temporal parameters to data list
    data_list_est <- append(
      data_list,
      list(
        rho1 = sampling_config$rho1[i + 1], # Spatial range parameter
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
    cat("    - Value of rho1 parameter:", sampling_config$rho1[i + 1], "\n")
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
      seed = i, # Fold-specific seed for reproducibility
      max_attempts = sampling_config$max_attempts,
      warmup_init = sampling_config$warmup_init,
      sampling_init = sampling_config$sampling_init,
      n_chains = sampling_config$n_chains,
      n_parallel_chains = sampling_config$n_parallel_chains,
      adapt_delta_init = sampling_config$adapt_delta_init
    )

    fit$save_object(file = file.path(output_path, paste0("fit_cv", i, ".rds")))
    
    # ========== DIAGNOSTIC GENERATION ==========
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
      y_transform = sampling_config$y_transform,
      adm1_code = rep(test_region, each = n2),
      adm1_name = rep(test_region_name, each = n2),
      datetime = rep(as.Date(df$Datetime[1:n2], tz = "GMT"), times = n1)
    )

    # Calculate performance metrics for this fold
    rMSE <- c(rMSE, sqrt(mean(ppc_diagnostics_all[[as.character(i + 1)]]$residual^2)))
    MAE <- c(MAE, mean(abs(ppc_diagnostics_all[[as.character(i + 1)]]$residual)))
  }

  # ========== FINAL SUMMARY AND REPORTING ==========
  cat("\n\n=============================================\n")
  cat("         FINAL RESULTS SUMMARY\n")
  cat("=============================================\n\n")

  # ========== CROSS-VALIDATION PERFORMANCE ==========
  # Display mean performance metrics across all folds
  cat("Model Performance:\n")
  cat("  Mean rMSE:", round(mean(rMSE), 4), "\n")
  cat("  Mean MAE:", round(mean(MAE), 4), "\n\n")

  # Display convergence status per fold
  cat("Convergence Summary Across Folds:\n")
  for (i in 1:length(convergence_results)) {
    cat(
      "  CV Fold", i, "- Max Rhat:",
      round(convergence_results[[as.character(i)]]$diagnostics$max_rhat, 4),
      "| Converged:", convergence_results[[as.character(i)]]$converged, "\n"
    )
  }

  # Create comprehensive summary table and save
  # Combines model performance and convergence diagnostics per fold
  # Au lieu de cv_fold = 0:n_folds, utilise les folds qui ont réellement été traités
  folds_completed <- as.numeric(names(convergence_results)) - 1

  final_summary <- data.frame(
    cv_fold = folds_completed,
    rMSE = rMSE,
    MAE = MAE,
    converged = sapply(convergence_results, function(x) x$converged),
    max_rhat = sapply(convergence_results, function(x) x$diagnostics$max_rhat)
  )

  write.csv(final_summary, file.path(output_path, "final_cv_summary.csv"), row.names = FALSE)

  # ========== BETA COEFFICIENT ANALYSIS ==========
  # Analyze feature coefficients only if features are present in the model
  if (length(selected_features) > 0) {
    cat("\n=============================================\n")
    cat("         BETA COEFFICIENTS SUMMARY\n")
    cat("=============================================\n\n")

    # Combine beta diagnostics from all folds
    all_beta_summary <- do.call(rbind, beta_diagnostics_all)
    write.csv(all_beta_summary,
      file.path(prior_posterior_path, "beta_summary_all_folds.csv"),
      row.names = FALSE
    )

    # Calculate average statistics across folds for each beta
    # Identifies which features have consistent effects across regions
    # Flags features whose credible intervals don't include zero
    beta_avg_summary <- all_beta_summary %>%
      group_by(feature, beta_name) %>%
      summarise(
        avg_posterior_mean = mean(posterior_mean), # Mean effect size
        avg_posterior_sd = mean(posterior_sd), # Uncertainty
        avg_prob_positive = mean(prob_positive), # Direction consistency
        times_includes_zero = sum(includes_zero), # Instances of weak signal
        avg_rhat = mean(rhat), # Convergence indicator
        min_ess_bulk = min(ess_bulk), # Effective samples
        .groups = "drop"
      )

    # Save averaged summaries
    write.csv(beta_avg_summary,
      file.path(prior_posterior_path, "beta_summary_averaged.csv"),
      row.names = FALSE
    )

    # Display feature coefficient summary
    cat("Beta Coefficients (averaged across folds):\n")
    print(beta_avg_summary, n = Inf)
  } else {
    # No features selected - this is an intercept-only GP model
    cat("\n=============================================\n")
    cat("         NO BETA COEFFICIENTS\n")
    cat("=============================================\n")
    cat("No features selected - skipping beta analysis\n")
  }

  # ========== POSTERIOR PREDICTIVE CHECK ==========
  # Aggregate prediction performance metrics across all folds
  all_ppc_summary <- do.call(rbind, ppc_diagnostics_all)
  write.csv(all_ppc_summary,
    file.path(prior_posterior_path, "ppc_summary_all_folds.csv"),
    row.names = FALSE
  )

  # Calculate posterior predictive check statistics
  # Coverage_95: Percent of observations within 95% credible interval
  # Low coverage suggests model is overconfident or misspecified
  cat("\n=============================================\n")
  cat("    POSTERIOR PREDICTIVE CHECK SUMMARY\n")
  cat("=============================================\n\n")

  ppc_stats <- all_ppc_summary %>%
    group_by(cv_fold) %>%
    summarise(
      n_observations = n(),
      coverage_95 = mean(in_ci_95) * 100, # Ideal: ~95%
      mean_abs_residual = mean(abs(residual)),
      rmse = sqrt(mean(residual^2)),
      .groups = "drop"
    )

  # Save PPC statistics
  write.csv(ppc_stats,
    file.path(prior_posterior_path, "ppc_statistics_by_fold.csv"),
    row.names = FALSE
  )

  # Display posterior predictive check results
  cat("Posterior Predictive Check Statistics:\n")
  print(ppc_stats, n = Inf)

  # ========== COMPLETION MESSAGE ==========
  cat("\n✓ All diagnostics saved!\n")
  cat("  - Convergence diagnostics:", diagnostics_path, "\n")
  cat("  - Beta and PPC diagnostics:", prior_posterior_path, "\n")

  # ========== RETURN RESULTS ==========
  # Return results invisibly to allow assignment but avoid printing
  # Allows downstream analysis of fit objects and diagnostics
  invisible(list(
    rMSE = rMSE,
    MAE = MAE,
    convergence_results = convergence_results,
    beta_diagnostics_all = beta_diagnostics_all,
    ppc_diagnostics_all = ppc_diagnostics_all,
    output_path = output_path,
    llkd = post_mean_llkd
  ))
}
