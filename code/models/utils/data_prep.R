# Preprocess training data with standardization for Stan modeling
# Filters data by regions, applies log-odds transformation to target variable,
# and standardizes covariates for use in Bayesian spatial models
prep_stan_data <- function(df_long, df_loc, regions, y_transform = TRUE, main_var = "fcs", features = NULL, standardize = TRUE) {
  # Filter data by selected regions
  df <- df_long[df_long$adm1_code %in% regions, ] # A MODIFIER

  # Number of unique regions/administrative units
  n1 <- length(regions)
  # Extract target variable
  y <- df[, main_var]
  # Apply logit transformation to map [0,1] to real numbers
  if (y_transform) {
    y <- log(y / (1 - y)) # Logit transformation for probability data
  }
  # Number of observations per region (time steps)
  n2 <- length(y) / n1
  # Extract spatial coordinates and convert from meters to kilometers
  loc <- df_loc[df_loc$adm1_code %in% regions, c("Easting", "Northing")] / 1000 # Spatial coordinates in kilometers
  # Initialize result list with basic spatial and target data
  res <- list(
    N1 = n1, # Number of regions
    N2 = n2, # Number of observations per region
    X1 = loc, # Spatial coordinates
    X2 = matrix(c(1:n2), n2, 1), # Time index for each observation
    y = y # Target variable (food consumption score or logit-transformed)
  )

  # Handle covariates/features if provided
  if (!is.null(features)) {
    # Convert selected features to matrix format
    covmat <- as.matrix(df[, features])

    # Initialize scaling parameters (mean and standard deviation)
    # These will be returned for use in standardizing test data
    cov_mean <- NULL
    cov_sd <- NULL

    if (standardize) {
      # Compute mean for each feature column
      cov_mean <- colMeans(covmat)
      # Compute standard deviation for each feature column
      cov_sd <- apply(covmat, 2, sd)
      # Prevent division by zero for constant columns (set sd to 1)
      cov_sd[cov_sd == 0] <- 1

      # Center and scale each column to mean=0, sd=1
      covmat <- scale(covmat, center = cov_mean, scale = cov_sd)
    }

    # Update result list to include features and scaling parameters
    res <- list(
      N1 = n1, # Number of regions
      N2 = n2, # Number of observations per region
      P = length(features), # Number of covariates
      X1 = loc, # Spatial coordinates
      X2 = matrix(c(1:n2), n2, 1), # Time index
      D = covmat, # Standardized covariate matrix
      y = y, # Target variable
      cov_mean = cov_mean, # Mean values for standardization (for test data)
      cov_sd = cov_sd # Standard deviation values (for test data)
    )
  }
  return(res)
}

prep_stan_data_names <- function(df_long, df_loc, regions_name, y_transform = TRUE, main_var = "fcs", features = NULL, standardize = TRUE) {
  # Filter data by selected regions
  df <- df_long[df_long$adm1_name %in% regions_name, ]

  # Number of unique regions/administrative units
  n1 <- length(regions_name)
  # Extract target variable
  y <- df[, main_var]
  # Apply logit transformation to map [0,1] to real numbers
  if (y_transform) {
    y <- log(y / (1 - y)) # Logit transformation for probability data
  }
  # Number of observations per region (time steps)
  n2 <- length(y) / n1
  # Extract spatial coordinates and convert from meters to kilometers
  loc <- df_loc[df_loc$adm1_name %in% regions_name, c("Easting", "Northing")] / 1000 # Spatial coordinates in kilometers
  # Initialize result list with basic spatial and target data
  res <- list(
    N1 = n1, # Number of regions
    N2 = n2, # Number of observations per region
    X1 = loc, # Spatial coordinates
    X2 = matrix(c(1:n2), n2, 1), # Time index for each observation
    y = y # Target variable (food consumption score or logit-transformed)
  )

  # Handle covariates/features if provided
  if (!is.null(features)) {
    # Reset index
    df <- df |> tibble::rowid_to_column()
    # Convert selected features to matrix format
    covmat <- as.matrix(df[, features])

    # Initialize scaling parameters (mean and standard deviation)
    # These will be returned for use in standardizing test data
    cov_mean <- NULL
    cov_sd <- NULL

    if (standardize) {
      # Compute mean for each feature column
      cov_mean <- colMeans(covmat)
      # Compute standard deviation for each feature column
      cov_sd <- apply(covmat, 2, sd)
      # Prevent division by zero for constant columns (set sd to 1)
      cov_sd[cov_sd == 0] <- 1

      # Center and scale each column to mean=0, sd=1
      covmat <- scale(covmat, center = cov_mean, scale = cov_sd)
    }

    # Update result list to include features and scaling parameters
    res <- list(
      N1 = n1, # Number of regions
      N2 = n2, # Number of observations per region
      P = length(features), # Number of covariates
      X1 = loc, # Spatial coordinates
      X2 = matrix(c(1:n2), n2, 1), # Time index
      D = covmat, # Standardized covariate matrix
      y = y, # Target variable
      cov_mean = cov_mean, # Mean values for standardization (for test data)
      cov_sd = cov_sd # Standard deviation values (for test data)
    )
  }
  return(res)
}

# Preprocess testing/prediction data using same standardization as training
# Applies training set scaling parameters to test data to ensure consistency
# in covariate standardization across train/test splits
# using code as reference
prep_stan_data_pred <- function(df_long, df_loc, regions, main_var = "fcs",
                                features = NULL, standardize = TRUE,
                                cov_mean = NULL, cov_sd = NULL) {
  # Filter data by selected regions
  df <- df_long[df_long$adm1_code %in% regions, ]
  # Extract target variable
  y <- df[, main_var]
  # Number of unique regions
  n1 <- length(regions)
  # Number of observations per region
  n2 <- length(y) / n1
  # Extract and scale spatial coordinates to kilometers
  loc <- df_loc[df_loc$adm1_code %in% regions, c("Easting", "Northing")] / 1000
  # Initialize result list with basic spatial and target data
  res <- list(list(n1 = n1, n2 = n2, x1_new = loc, x2_new = matrix(c(1:n2), n2, 1)), y)

  if (!is.null(features)) {
    # Convert selected features to matrix format
    covmat <- as.matrix(df[, features])

    if (standardize) {
      # CRITICAL: Use the SAME mean and sd from training data
      # This ensures test data is standardized consistently with training data
      if (is.null(cov_mean) || is.null(cov_sd)) {
        stop("When standardize=TRUE, must provide cov_mean and cov_sd from training data")
      }

      # Apply training standardization parameters to test data
      covmat <- scale(covmat, center = cov_mean, scale = cov_sd)
    }

    # Update result list to include standardized covariates
    res <- list(list(
      n1 = n1, # Number of regions
      n2 = n2, # Number of observations per region
      x1_new = loc, # Spatial coordinates
      x2_new = matrix(c(1:n2), n2, 1), # Time index
      D_new = covmat # Standardized covariate matrix
    ), y)
  }
  return(res)
}

# using names as reference
prep_stan_data_pred_name <- function(df_long, df_loc, regions_name, main_var = "fcs",
                                     features = NULL, standardize = TRUE,
                                     cov_mean = NULL, cov_sd = NULL) {
  # Filter data by selected regions
  df <- df_long[df_long$adm1_name %in% regions_name, ]
  # Extract target variable
  y <- df[, main_var]
  # Number of unique regions
  n1 <- length(regions_name)
  # Number of observations per region
  n2 <- length(y) / n1
  # Extract and scale spatial coordinates to kilometers
  loc <- df_loc[df_loc$adm1_name %in% regions_name, c("Easting", "Northing")] / 1000
  # Initialize result list with basic spatial and target data
  res <- list(list(n1 = n1, n2 = n2, x1_new = loc, x2_new = matrix(c(1:n2), n2, 1)), y)

  if (!is.null(features)) {
    # Convert selected features to matrix format
    covmat <- as.matrix(df[, features])

    if (standardize) {
      # CRITICAL: Use the SAME mean and sd from training data
      # This ensures test data is standardized consistently with training data
      if (is.null(cov_mean) || is.null(cov_sd)) {
        stop("When standardize=TRUE, must provide cov_mean and cov_sd from training data")
      }

      # Apply training standardization parameters to test data
      covmat <- scale(covmat, center = cov_mean, scale = cov_sd)
    }

    # Update result list to include standardized covariates
    res <- list(list(
      n1 = n1, # Number of regions
      n2 = n2, # Number of observations per region
      x1_new = loc, # Spatial coordinates
      x2_new = matrix(c(1:n2), n2, 1), # Time index
      D_new = covmat # Standardized covariate matrix
    ), y)
  }
  return(res)
}

# Create and engineer features: apply transformations and interactions
# This function prepares raw variables for modeling by creating derived features,
# log transformations, and interaction terms
create_features <- function(df, country) {
  if (country == "Chad") {
    df$Ramadan_past90days_Muslimpop <- df$Ramadan_past90days * df$Muslim
    return(df)
  }
  if (country == "Nigeria") {
    # Normalize Muslim population percentage from 0-100 to 0-1 scale
    df$Muslim <- df$Muslim / 100

    # Create interaction features
    # Ramadan effect interaction with Muslim population proportion
    df$Ramadan_past90days_Muslimpop <- df$Ramadan_past90days * df$Muslim
    # Food inflation interaction with Multidimensional Poverty Index
    df$food_inflation_MPI <- df$food_inflation_cubic_intp * df$MPI
    # Log food inflation interaction with poverty index
    df$log_food_inflation_MPI <- log(df$food_inflation_cubic_intp) * df$MPI

    # Apply log transformations to right-skewed variables (add 1 to handle zeros)
    df$log_food_inflation <- log(df$food_inflation_cubic_intp)
    df$log_fatalities_battles <- log(df$n_fatalities_Battles_rolsum_90days + 1)
    df$log_fatalities_violenceagainstcivilians <- log(df$n_fatalities_Violence_against_civilians_rolsum_90days + 1)
    df$log_fatalities_explosions <- log(df$n_fatalities_Explosions.Remote_violence_rolsum_90days + 1)
    df$log_Area <- log(df$Area + 1)
    df$log_currency_exchange <- log(df$currency_exchange + 1)

    # Data type conversions
    # Ensure day_num is numeric (handles day of year)
    df$day_num <- as.numeric(df$day_num)

    return(df)
  }
}


# Save train and test datasets for cross-validation split
# Splits data by region for leave-region-out cross-validation,
# combines covariates with spatial coordinates, and saves to CSV files
save_data_set <- function(df_90, df_loc, i, selected_features,
                          path_data_train = NULL, path_data_test = NULL) {
  # Extract test regions for fold i (leave-region-out cross-validation)
  tes_region <- df_90$adm1_name[df_90$cv_ind == i] |> unique()

  # Extract training regions (all regions except test fold)
  train_region <- df_90$adm1_name[df_90$cv_ind != i] |> unique()

  # Prepare training covariates and target variable
  df_train_covariables <- df_90 |>
    filter(adm1_name %in% train_region, cv_ind != i) |>
    select(Datetime, adm1_name, fcs, cv_ind, all_of(selected_features))

  # Extract and standardize training spatial coordinates
  df_train_loc <- df_loc |>
    filter(adm1_name %in% train_region) |>
    select(adm1_name, Easting, Northing) |>
    mutate(Easting = Easting / 1000, Northing = Northing / 1000) # Convert to kilometers

  # Merge training location data with covariates
  df_train <- df_train_loc |>
    full_join(df_train_covariables, by = "adm1_name")

  # Sort training data by region name and datetime for consistent ordering
  df_train <- df_train[order(df_train$adm1_name, df_train$Datetime), ]

  # Prepare testing covariates and target variable
  df_test_covariables <- df_90 |>
    filter(adm1_name %in% tes_region, cv_ind == i) |>
    select(Datetime, adm1_name, fcs, cv_ind, all_of(selected_features))

  # Extract and standardize testing spatial coordinates
  df_test_loc <- df_loc |>
    filter(adm1_name %in% tes_region) |>
    select(adm1_name, Easting, Northing) |>
    mutate(Easting = Easting / 1000, Northing = Northing / 1000) # Convert to kilometers

  # Merge testing location data with covariates
  df_test <- df_test_loc |>
    full_join(df_test_covariables, by = "adm1_name")

  # Sort testing data by region name and datetime for consistent ordering
  df_test <- df_test[order(df_test$adm1_name, df_test$Datetime), ]

  # Save datasets to CSV if file paths are provided
  if (!is.null(path_data_train) && !is.null(path_data_test)) {
    # Construct output file paths with cross-validation fold index
    train_path <- paste0(path_data_train, "_cv", i, ".csv")
    test_path <- paste0(path_data_test, "_cv", i, ".csv")

    # Write datasets to CSV files without row names
    write.csv(df_train, train_path, row.names = FALSE)
    write.csv(df_test, test_path, row.names = FALSE)

    # Print confirmation message with saved file paths
    cat("Data saved:\n")
    cat("  Train:", train_path, "\n")
    cat("  Test:", test_path, "\n")
  }

  return()
}


# Save predict dataset
# combines covariates with spatial coordinates, and saves to CSV files
save_data_set_predict <- function(df_90, df_loc, selected_features, path_data_pred = NULL) {
  # Extract test regions for fold i (leave-region-out cross-validation)
  tes_region <- df_90 |>
    pull(adm1_name) |>
    unique()

  # Prepare testing covariates and target variable
  df_test_covariables <- df_90 |>
    filter(adm1_name %in% tes_region) |>
    select(Datetime, adm1_name, fcs, all_of(selected_features))

  # Extract and standardize testing spatial coordinates
  df_test_loc <- df_loc |>
    filter(adm1_name %in% tes_region) |>
    select(adm1_name, Easting, Northing) |>
    mutate(Easting = Easting / 1000, Northing = Northing / 1000) # Convert to kilometers

  # Merge testing location data with covariates
  df_test <- df_test_loc |>
    full_join(df_test_covariables, by = "adm1_name")

  # Sort testing data by region name and datetime for consistent ordering
  df_test <- df_test[order(df_test$adm1_name, df_test$Datetime), ]

  # Save datasets to CSV if file paths are provided
  if (!is.null(path_data_pred)) {
    # Construct output file paths with cross-validation fold index
    test_path <- paste0(path_data_pred, ".csv")
    write.csv(df_test, test_path, row.names = FALSE)

    # Print confirmation message with saved file paths
    cat("Data saved:\n")
    cat("  Test:", test_path, "\n")
  }

  return()
}
