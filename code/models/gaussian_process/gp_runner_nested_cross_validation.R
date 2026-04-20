# Nested cross-validation runner for GP models.
# Implements run_gp_model_cv(): outer loop tests one region at a time,
# inner loop tunes rho1 over a grid on the remaining regions.

run_gp_model_cv <- function(
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
  library(tidyr)
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
  # This ensures we have proper survey response annotations for each test fold
  if (country == "Nigeria") {
    region_90 <- df$adm1_code[df$data_type == "SURVEY 90 days"] |> unique()
    region_90_name <- df$adm1_name[df$data_type == "SURVEY 90 days"] |> unique()
    df <- df[df$data_type == "SURVEY 90 days", ]
  }

  # ========== STAN MODEL COMPILATION ==========
  # Load and compile the Stan probabilistic model
  # The Stan model defines the likelihood, priors, and parameters to estimate
  stan_path <- file.path(topdir, "code", "models", "stan")
  stan_file <- file.path(stan_path, stan_file_name)
  mod <- cmdstanr::cmdstan_model(stan_file, include_paths = stan_path)

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
  for (i in 0:n_folds) {
    cat("\n\n=============================================\n")
    cat("         CROSS-VALIDATION FOLD (FIRST LAYER) ", i, "\n")
    cat("=============================================\n\n")

    # ========== DATA SPLIT ==========

    # Définir la région de test pour ce fold
    test_region <- unique(df$adm1_code[df$cv_ind == i])
    test_region_name <- unique(df$adm1_name[df$cv_ind == i])
    # Définir la région train pour ce fold
    train_region <- unique(df$adm1_code[df$cv_ind != i])

    cat("  Test Region:", test_region_name, "(cv_ind =", i, ")\n")
    cat("  Training Regions:", length(train_region), "regions\n")

    # Créer les folds de base à partir des données d'entraînement uniquement
    # (on exclut la région de test)
    cv_folds <- list()
    available_cv_ind <- unique(df$cv_ind[df$cv_ind != i])

    # indice qui map les index de chaque cv fold
    for (cv_idx in available_cv_ind) {
      cv_folds[[as.character(cv_idx)]] <- which(df$cv_ind == cv_idx)
    }
    cat("  Available CV folds for nested CV:", paste(names(cv_folds), collapse = ", "), "\n")

    # Train/Validation sets: identifier les cv_ind disponibles (sans le test fold)
    cv_indices <- sort(as.numeric(names(cv_folds)))

    # Créer les combinaisons train/validation (4 splits possibles)
    train_fold <- list()
    val_fold <- list()

    if (i == 1) {
      # test_fold = 1 (on a les folds 0,2,3,4,5 disponibles)
      train_fold[[1]] <- c(cv_folds[["2"]], cv_folds[["4"]], cv_folds[["5"]], cv_folds[["0"]])
      val_fold[[1]] <- cv_folds[["3"]]

      train_fold[[2]] <- c(cv_folds[["2"]], cv_folds[["3"]], cv_folds[["5"]], cv_folds[["0"]])
      val_fold[[2]] <- cv_folds[["4"]]

      train_fold[[3]] <- c(cv_folds[["2"]], cv_folds[["3"]], cv_folds[["4"]], cv_folds[["0"]])
      val_fold[[3]] <- cv_folds[["5"]]

      train_fold[[4]] <- c(cv_folds[["3"]], cv_folds[["4"]], cv_folds[["5"]], cv_folds[["0"]])
      val_fold[[4]] <- cv_folds[["2"]]

      train_fold[[5]] <- c(cv_folds[["2"]], cv_folds[["3"]], cv_folds[["4"]], cv_folds[["5"]])
      val_fold[[5]] <- cv_folds[["0"]]
    } else if (i == 2) {
      train_fold[[1]] <- c(cv_folds[["1"]], cv_folds[["3"]], cv_folds[["5"]], cv_folds[["0"]])
      val_fold[[1]] <- cv_folds[["4"]]

      train_fold[[2]] <- c(cv_folds[["1"]], cv_folds[["4"]], cv_folds[["5"]], cv_folds[["0"]])
      val_fold[[2]] <- cv_folds[["3"]]

      train_fold[[3]] <- c(cv_folds[["1"]], cv_folds[["3"]], cv_folds[["4"]], cv_folds[["0"]])
      val_fold[[3]] <- cv_folds[["5"]]

      train_fold[[4]] <- c(cv_folds[["0"]], cv_folds[["3"]], cv_folds[["4"]], cv_folds[["5"]])
      val_fold[[4]] <- cv_folds[["1"]]

      train_fold[[5]] <- c(cv_folds[["1"]], cv_folds[["3"]], cv_folds[["4"]], cv_folds[["5"]])
      val_fold[[5]] <- cv_folds[["0"]]
    } else if (i == 3) {
      # test_fold = 3
      train_fold[[1]] <- c(cv_folds[["1"]], cv_folds[["2"]], cv_folds[["5"]], cv_folds[["0"]])
      val_fold[[1]] <- cv_folds[["4"]]

      train_fold[[2]] <- c(cv_folds[["1"]], cv_folds[["2"]], cv_folds[["4"]], cv_folds[["0"]])
      val_fold[[2]] <- cv_folds[["5"]]

      train_fold[[3]] <- c(cv_folds[["1"]], cv_folds[["0"]], cv_folds[["4"]], cv_folds[["5"]])
      val_fold[[3]] <- cv_folds[["2"]]

      train_fold[[4]] <- c(cv_folds[["4"]], cv_folds[["2"]], cv_folds[["5"]], cv_folds[["0"]])
      val_fold[[4]] <- cv_folds[["1"]]

      train_fold[[5]] <- c(cv_folds[["1"]], cv_folds[["2"]], cv_folds[["4"]], cv_folds[["5"]])
      val_fold[[5]] <- cv_folds[["0"]]
    } else if (i == 4) {
      # test_fold = 4
      train_fold[[1]] <- c(cv_folds[["1"]], cv_folds[["2"]], cv_folds[["3"]], cv_folds[["0"]])
      val_fold[[1]] <- cv_folds[["5"]]

      train_fold[[2]] <- c(cv_folds[["1"]], cv_folds[["3"]], cv_folds[["5"]], cv_folds[["0"]])
      val_fold[[2]] <- cv_folds[["2"]]

      train_fold[[3]] <- c(cv_folds[["1"]], cv_folds[["2"]], cv_folds[["5"]], cv_folds[["0"]])
      val_fold[[3]] <- cv_folds[["3"]]

      train_fold[[4]] <- c(cv_folds[["0"]], cv_folds[["2"]], cv_folds[["3"]], cv_folds[["5"]])
      val_fold[[4]] <- cv_folds[["1"]]

      train_fold[[5]] <- c(cv_folds[["1"]], cv_folds[["2"]], cv_folds[["3"]], cv_folds[["5"]])
      val_fold[[5]] <- cv_folds[["0"]]
    } else if (i == 5) {
      # test_fold = 5
      train_fold[[1]] <- c(cv_folds[["0"]], cv_folds[["2"]], cv_folds[["3"]], cv_folds[["4"]])
      val_fold[[1]] <- cv_folds[["1"]]

      train_fold[[2]] <- c(cv_folds[["1"]], cv_folds[["3"]], cv_folds[["4"]], cv_folds[["0"]])
      val_fold[[2]] <- cv_folds[["2"]]

      train_fold[[3]] <- c(cv_folds[["1"]], cv_folds[["2"]], cv_folds[["4"]], cv_folds[["0"]])
      val_fold[[3]] <- cv_folds[["3"]]

      train_fold[[4]] <- c(cv_folds[["1"]], cv_folds[["2"]], cv_folds[["3"]], cv_folds[["0"]])
      val_fold[[4]] <- cv_folds[["4"]]

      train_fold[[5]] <- c(cv_folds[["1"]], cv_folds[["2"]], cv_folds[["3"]], cv_folds[["4"]])
      val_fold[[5]] <- cv_folds[["0"]]
    } else if (i == 0) {
      # test_fold = 0
      train_fold[[1]] <- c(cv_folds[["1"]], cv_folds[["3"]], cv_folds[["4"]], cv_folds[["5"]])
      val_fold[[1]] <- cv_folds[["2"]]

      train_fold[[2]] <- c(cv_folds[["1"]], cv_folds[["2"]], cv_folds[["4"]], cv_folds[["5"]])
      val_fold[[2]] <- cv_folds[["3"]]

      train_fold[[3]] <- c(cv_folds[["1"]], cv_folds[["2"]], cv_folds[["3"]], cv_folds[["5"]])
      val_fold[[3]] <- cv_folds[["4"]]

      train_fold[[4]] <- c(cv_folds[["2"]], cv_folds[["3"]], cv_folds[["4"]], cv_folds[["5"]])
      val_fold[[4]] <- cv_folds[["1"]]

      train_fold[[5]] <- c(cv_folds[["1"]], cv_folds[["2"]], cv_folds[["3"]], cv_folds[["4"]])
      val_fold[[5]] <- cv_folds[["5"]]
    }


    # Sauvegarder les 4 splits train/validation pour ce fold
    list_cv <- list()
    for (j in 1:5) {
      list_cv[[j]] <- list(
        train = train_fold[[j]],
        val = val_fold[[j]]
      )
    }

    # Sauvegarder dans un fichier RDS (équivalent de pickle en R)
    saveRDS(
      list_cv,
      file = file.path(output_path, paste0("list_cv_fold", i, ".rds"))
    )

    cat("  Created", length(list_cv), "nested CV splits for hyperparameter tuning\n")
    cat("  Saved to: list_cv_fold", i, ".rds\n\n")

    y_transform <- sampling_config$y_transform

    for (j in 1:5) {
      cat("\n\n=============================================\n")
      cat("         CROSS-VALIDATION FOLD (SECOND LAYER) ", j, "\n")
      cat("=============================================\n\n")

      # ========== DATA STANDARDIZATION ==========
      # Prepare training data: apply logit transform to target variable (FCS)
      # Standardize covariates to unit variance (zero mean, sd=1)
      # This prevents data leakage and improves MCMC sampling efficiency
      df_train_val <- df[c(list_cv[[j]]$train, list_cv[[j]]$val), ]

      # train and val region code
      train_region <- unique(df[list_cv[[j]]$train, "adm1_name"])
      val_region <- unique(df[list_cv[[j]]$val, "adm1_name"])

      # train and val region names
      train_region_name <- unique(df[list_cv[[j]]$train, "adm1_name"])
      val_region_name <- unique(df[list_cv[[j]]$val, "adm1_name"])
      test_region_name <- df |>
        filter(cv_ind == i) |>
        pull(adm1_name) |>
        unique()

      data_list_tr <- prep_stan_data_names(
        df_long = df_train_val,
        df_loc = df_loc,
        regions = train_region,
        y_transform = y_transform, # Apply logit(FCS)
        features = selected_features,
        standardize = TRUE # Center and scale covariates
      )

      # Extract standardization parameters computed from TRAINING data
      # These will be applied to test data to maintain consistency
      cov_mean <- data_list_tr$cov_mean
      cov_sd <- data_list_tr$cov_sd

      # Prepare test data using TRAINING set's standardization parameters
      # Prevents test data information leakage into model fitting
      data_tes_process <- prep_stan_data_pred_name(
        df_long = df_train_val,
        df_loc = df_loc,
        regions = val_region,
        features = selected_features,
        standardize = TRUE,
        cov_mean = cov_mean,
        cov_sd = cov_sd
      )


      # Combine train and test data lists for Stan sampling
      data_list <- append(data_list_tr, data_tes_process[[1]])

      n1 <- data_tes_process[[1]]$n1
      n2 <- data_tes_process[[1]]$n2

      # ========== GP DATA SIZE SUMMARY ==========
      # Display Gaussian Process data dimensions

      cat("  Gaussian Process Data Size:\n")
      cat("    - Regions in the train ", train_region_name, "\n")
      cat("    - Regions in the val :", val_region_name, "\n")
      cat("    - Regions in the test:", test_region_name, "\n")

      if (length(intersect(train_region_name, c(val_region_name, test_region_name))) > 0) {
        cat("Erreur: les region train, val et test sont mal séparées")
      }

      cat("    - Spatial locations in the train set (N1):", data_list_tr$N1, "\n")
      cat("    - Spatial locations in the val set (n1):", data_tes_process[[1]]$n1, "\n")
      cat("    - Temporal observations per location in the train (N2):", data_list$N2, "\n")
      cat("    - Temporal observations per location in the test (n2):", data_tes_process[[1]]$n2, "\n")
      cat("    - Total observations in the train:", data_list_tr$N1 * data_list$N2, "\n")
      cat("    - Total observations in the val:", data_tes_process[[1]]$n1 * data_tes_process[[1]]$n2, "\n")

      cat("\n")

      # ========== STAN SAMPLING WITH ADAPTIVE CONVERGENCE ==========
      for (rho1_cv in sampling_config$rho1) {
        # Add spatial/temporal parameters to data list
        data_list_est <- append(
          data_list,
          list(
            rho1 = rho1_cv, # Spatial range parameter
            Hurst2 = sampling_config$Hurst2 # Matérn smoothness
          )
        )
        # Fit the model using adaptive MCMC sampling
        # Automatically adjusts sampler parameters if convergence criteria not met
        # print(as.numeric(paste0(i, j, rho1_cv)))
        fit <- adaptive_sample(
          mod,
          data_list_est,
          seed = as.integer(paste0(i, j, rho1_cv)), # Fold-specific seed for reproducibility
          max_attempts = sampling_config$max_attempts,
          warmup_init = sampling_config$warmup_init,
          sampling_init = sampling_config$sampling_init,
          n_chains = sampling_config$n_chains,
          n_parallel_chains = sampling_config$n_parallel_chains,
          adapt_delta_init = sampling_config$adapt_delta_init
        )

        # ========== DIAGNOSTIC GENERATION ==========
        # Check convergence criteria: Rhat < 1.05, ESS_bulk > 400, no divergences
        convergence_results[[paste(i, j, rho1_cv, sep = "_")]] <- check_convergence(
          fit,
          i,
          diagnostics_path,
          write_res = FALSE,
          cv_fold_nested = j,
          rho1_cv = rho1_cv
        )

        # ========== MODEL EVALUATION ==========
        # Extract posterior predictive samples and compute prediction intervals
        # Calculate residuals: y_test - predicted values
        # Assess 95% CI coverage and prediction accuracy
        ppc_diagnostics_all[[paste(i, j, rho1_cv, sep = "_")]] <- extract_posterior_predictive_data(
          fit = fit,
          y_true = data_tes_process[[2]],
          cv_fold = i,
          prior_posterior_path = prior_posterior_path,
          n1 = n1,
          n2 = n2,
          y_transform = y_transform,
          adm1_code = rep(val_region, each = n2),
          adm1_name = rep(val_region_name, each = n2),
          datetime = rep(as.Date(df$Datetime[1:n2], tz = "GMT"), times = n1),
          write_res = FALSE,
          cv_fold_nested = j,
          rho1_cv = rho1_cv
        )

        # Calculate performance metrics for this fold
        rMSE[[paste(i, j, rho1_cv, sep = "_")]] <- sqrt(mean(ppc_diagnostics_all[[paste(i, j, rho1_cv, sep = "_")]]$residual^2))
        MAE[[paste(i, j, rho1_cv, sep = "_")]] <- mean(abs(ppc_diagnostics_all[[paste(i, j, rho1_cv, sep = "_")]]$residual))
      }
    }
  }

  # ========== FINAL SUMMARY AND REPORTING ==========
  cat("\n\n=============================================\n")
  cat("         FINAL RESULTS SUMMARY\n")
  cat("=============================================\n\n")

  # Create comprehensive summary table and save
  # Combines model performance and convergence diagnostics per fold
  final_summary <- data.frame(
    cv_fold = names(rMSE),
    rMSE = unlist(rMSE),
    MAE = unlist(MAE),
    converged = unlist(sapply(convergence_results, function(x) x$converged))
  )

  # Puis extraire les composants de la clé si tu veux les colonnes séparées
  final_summary <- final_summary |>
    separate(cv_fold, into = c("cv_fold_i", "cv_fold_j", "rho1"), sep = "_", convert = TRUE)

  write.csv(final_summary, file.path(output_path, "final_cv_summary.csv"), row.names = FALSE)

  # ========== RETURN RESULTS ==========
  # Return results invisibly to allow assignment but avoid printing
  # Allows downstream analysis of fit objects and diagnostics
  invisible(list(
    rMSE = rMSE,
    MAE = MAE,
    convergence_results = convergence_results,
    beta_diagnostics_all = beta_diagnostics_all,
    ppc_diagnostics_all = ppc_diagnostics_all,
    output_path = output_path
  ))
}
