# BYM2 benchmark runner
#
# Fits the spatio-temporal BYM2 model (code/models/stan/BYM2_ridge_prior.stan)
# with the same cross-validation folds, target transformation (logit FCS),
# covariates and train-set standardisation as the GP pipeline
# (code/models/gaussian_process/gp_runner_lib.R), so that the outputs can be
# compared directly (ppc_summary_all_folds.csv has the same format).
#
# The adjacency graph is read from data/new/<country>/<country>-graph-*.csv,
# produced by code/models/bym/build_adjacency.py.

# Scaling factor of the BYM2 model (Riebler et al., 2016): geometric mean of
# the marginal variances of the ICAR field under the sum-to-zero constraint.
# For a connected graph these variances are the diagonal of the Moore-Penrose
# inverse of Q, which equals solve(Q + J / n) - J / n with J the matrix of ones.
bym2_scaling_factor <- function(n_nodes, node1, node2) {
  W <- matrix(0, n_nodes, n_nodes)
  W[cbind(node1, node2)] <- 1
  W[cbind(node2, node1)] <- 1
  Q <- diag(rowSums(W)) - W
  J_n <- matrix(1 / n_nodes, n_nodes, n_nodes)
  Sigma <- solve(Q + J_n) - J_n
  exp(mean(log(diag(Sigma))))
}

# Read the graph (nodes and edges) of a country and compute the scaling factor
load_graph <- function(country, topdir) {
  cname <- tolower(country)
  data_dir <- file.path(topdir, "data", "new", cname)
  nodes <- read.csv(file.path(data_dir, paste0(cname, "-graph-nodes.csv")))
  edges <- read.csv(file.path(data_dir, paste0(cname, "-graph-edges.csv")))
  nodes <- nodes[order(nodes$node_id), ]
  list(
    nodes = nodes,
    edges = edges,
    scaling_factor = bym2_scaling_factor(nrow(nodes), edges$node1, edges$node2)
  )
}

# Map each row of a data frame to its graph node and time index
make_indices <- function(df_rows, nodes, dates) {
  area <- match(df_rows$adm1_name, nodes$adm1_name)
  time <- match(as.character(df_rows$Datetime), dates)
  if (anyNA(area)) {
    stop("Regions missing from the graph: ", paste(unique(df_rows$adm1_name[is.na(area)]), collapse = ", "))
  }
  if (anyNA(time)) stop("Some dates are missing from the time index")
  list(area = area, time = time)
}

# Trace plots of the hyperparameters and of beta (the full set of
# random effects is too large for the GP plotting function)
create_bym_convergence_plots <- function(fit, cv_fold, diagnostics_path, selected_features) {
  hyper <- c("b0", "sigma_s", "rho", "sigma_nu", "sigma_xi", "sigma")
  if (length(selected_features) > 0) hyper <- c(hyper, "kappa")
  draws <- fit$draws(variables = hyper)
  p_trace <- bayesplot::mcmc_trace(draws, facet_args = list(ncol = 2)) +
    ggtitle(paste("Trace Plots - CV Fold", cv_fold)) +
    theme_minimal()
  ggsave(file.path(diagnostics_path, paste0("trace_plot_cv", cv_fold, ".png")),
    p_trace,
    width = 12, height = 8, dpi = 300
  )

  if (length(selected_features) > 0) {
    p_beta <- bayesplot::mcmc_trace(fit$draws("beta"), facet_args = list(ncol = 3)) +
      ggtitle(paste("Beta Trace Plots - CV Fold", cv_fold)) +
      theme_minimal()
    ggsave(file.path(diagnostics_path, paste0("trace_plot_beta_cv", cv_fold, ".png")),
      p_beta,
      width = 14, height = 10, dpi = 300
    )
  }

  hyper_summary <- fit$summary(variables = hyper)
  write.csv(hyper_summary,
    file.path(diagnostics_path, paste0("hyperparameter_summary_cv", cv_fold, ".csv")),
    row.names = FALSE
  )
}

run_bym_model <- function(
  country,
  iso3,
  n_folds,
  sampling_config,
  model_config,
  selected_features,
  stan_file_name = "BYM2_ridge_prior.stan",
  topdir = here::here()
) {
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(bayesplot)
  library(cmdstanr)
  library(posterior)
  library(here)

  source(file.path(topdir, "code", "models", "utils", "convergence.R"))
  source(file.path(topdir, "code", "models", "utils", "data_prep.R"))
  source(file.path(topdir, "code", "models", "utils", "diagnostic.R"))

  # ========== STAN MODEL COMPILATION ==========
  stan_path <- file.path(topdir, "code", "models", "stan")
  mod <- cmdstanr::cmdstan_model(file.path(stan_path, stan_file_name), include_paths = stan_path)

  # ========== DATA LOADING AND PREPROCESSING ==========
  # Identical to run_gp_model
  cname <- tolower(country)
  df <- read.csv(
    file.path(topdir, "data", "new", cname, paste0(cname, "-weekly-with-features.csv")),
    header = TRUE
  )
  df <- df[order(df$adm1_name, df$Datetime), ]
  df_loc <- read.csv(
    file.path(topdir, "data", "new", cname, paste0(cname, "-static-data.csv")),
    header = TRUE
  )
  df <- create_features(df, country)
  missing_features <- setdiff(selected_features, names(df))
  if (length(missing_features) > 0) {
    stop("Missing features after preprocessing: ", paste(missing_features, collapse = ", "))
  }
  df_90 <- df[df$data_type == "SURVEY 90 days", ]

  # ========== GRAPH AND TIME INDEX ==========
  graph <- load_graph(country, topdir)
  dates <- sort(unique(as.character(df$Datetime)))
  cat("Graph:", nrow(graph$nodes), "nodes,", nrow(graph$edges), "edges,",
      "scaling factor =", round(graph$scaling_factor, 4), "\n")

  # ========== OUTPUT DIRECTORY STRUCTURE ==========
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  model_variant <- if (length(selected_features) > 0) "bym2_with_features" else "bym2"
  output_path <- file.path(
    topdir, "output", "bym2",
    paste0(iso3, "_", model_variant, "_", model_config$prior_type, "_", timestamp)
  )
  diagnostics_path <- file.path(output_path, "convergence_diagnostics")
  prior_posterior_path <- file.path(output_path, "prior_posterior_diagnostics")
  for (p in c(output_path, diagnostics_path, prior_posterior_path)) {
    if (!dir.exists(p)) dir.create(p, recursive = TRUE)
  }

  rMSE <- c()
  MAE <- c()
  convergence_results <- list()
  beta_diagnostics_all <- list()
  ppc_diagnostics_all <- list()

  # ========== CROSS-VALIDATION LOOP ==========
  # Same folds as the GP: all regions with cv_ind == i are held out together
  for (i in 0:n_folds) {
    cat("\n\n=============================================\n")
    cat("         CROSS-VALIDATION FOLD", i, "\n")
    cat("=============================================\n\n")

    # ========== DATA SPLIT AND STANDARDISATION ==========
    # The row filters below are the ones used inside prep_stan_data*, so that
    # y, D and the indices are aligned row by row
    if (country == "Nigeria") {
      train_region <- df_90$adm1_code[df_90$cv_ind != i] |> unique()
      test_region <- df_90$adm1_code[df_90$cv_ind == i] |> unique()
      if (length(test_region) == 0) next

      data_tr <- prep_stan_data(
        df_90, df_loc, train_region,
        y_transform = sampling_config$y_transform,
        features = selected_features,
        standardize = TRUE
      )
      data_te <- prep_stan_data_pred(
        df_90, df_loc, test_region,
        features = selected_features,
        standardize = TRUE,
        cov_mean = data_tr$cov_mean,
        cov_sd = data_tr$cov_sd
      )
      df_train <- df_90[df_90$adm1_code %in% train_region, ]
      df_test <- df_90[df_90$adm1_code %in% test_region, ]
    } else if (country == "Chad") {
      train_region_name <- df$adm1_name[df$cv_ind != i] |> unique()
      test_region_name <- df$adm1_name[df$cv_ind == i] |> unique()
      if (length(test_region_name) == 0) next

      data_tr <- prep_stan_data_names(
        df_long = df,
        df_loc = df_loc,
        regions_name = train_region_name,
        y_transform = sampling_config$y_transform,
        features = selected_features,
        standardize = TRUE
      )
      data_te <- prep_stan_data_pred_name(
        df, df_loc, test_region_name,
        features = selected_features,
        standardize = TRUE,
        cov_mean = data_tr$cov_mean,
        cov_sd = data_tr$cov_sd
      )
      df_train <- df[df$adm1_name %in% train_region_name, ]
      df_test <- df[df$adm1_name %in% test_region_name, ]
    }

    idx_tr <- make_indices(df_train, graph$nodes, dates)
    idx_te <- make_indices(df_test, graph$nodes, dates)
    y_test <- data_te[[2]]
    N <- length(data_tr$y)
    n <- nrow(df_test)
    P <- length(selected_features)

    data_list <- list(
      NS = nrow(graph$nodes),
      NT = length(dates),
      P = P,
      N = N,
      y = as.numeric(data_tr$y),
      D = if (P > 0) as.matrix(data_tr$D) else matrix(0, N, 0),
      area = idx_tr$area,
      time = idx_tr$time,
      n = n,
      D_new = if (P > 0) as.matrix(data_te[[1]]$D_new) else matrix(0, n, 0),
      area_new = idx_te$area,
      time_new = idx_te$time,
      N_edges = nrow(graph$edges),
      node1 = graph$edges$node1,
      node2 = graph$edges$node2,
      scaling_factor = graph$scaling_factor,
      pc_u = model_config$pc_u,
      pc_alpha = model_config$pc_alpha
    )

    saveRDS(data_tr$cov_mean, file = file.path(output_path, paste0("cov_mean_fold_", i, ".rds")))
    saveRDS(data_tr$cov_sd, file = file.path(output_path, paste0("cov_sd_fold_", i, ".rds")))

    cat("  BYM2 Data Size:\n")
    cat("    - Regions in the train set:", length(unique(idx_tr$area)), "\n")
    cat("    - Regions in the test set:", paste(unique(df_test$adm1_name), collapse = ", "), "\n")
    cat("    - Regions without data (prior and neighbours only):",
        nrow(graph$nodes) - length(unique(c(idx_tr$area))), "\n")
    cat("    - Observations train / test:", N, "/", n, "\n")
    cat("    - Number of features:", P, "\n\n")

    # ========== STAN SAMPLING WITH ADAPTIVE CONVERGENCE ==========
    fit <- adaptive_sample(
      mod,
      data_list,
      seed = i,
      max_attempts = sampling_config$max_attempts,
      warmup_init = sampling_config$warmup_init,
      sampling_init = sampling_config$sampling_init,
      n_chains = sampling_config$n_chains,
      n_parallel_chains = sampling_config$n_parallel_chains,
      adapt_delta_init = sampling_config$adapt_delta_init
    )
    fit$save_object(file = file.path(output_path, paste0("fit_cv", i, ".rds")))

    # ========== DIAGNOSTICS ==========
    convergence_results[[as.character(i + 1)]] <- check_convergence(fit, i, diagnostics_path)
    create_bym_convergence_plots(fit, i, diagnostics_path, selected_features)

    if (P > 0) {
      beta_diagnostics_all[[as.character(i + 1)]] <- extract_beta_diagnostics(
        fit = fit,
        cv_fold = i,
        prior_posterior_path = prior_posterior_path,
        selected_features = selected_features
      )
    }
    extract_posterior_mean(fit = fit, cv_fold = i, output_path = output_path)
    extract_posterior_mean_llkd(fit = fit, cv_fold = i, output_path = output_path)

    # ========== MODEL EVALUATION ==========
    # y_new is ordered as df_test (region by region, then by date), as y_test
    n1 <- length(unique(df_test$adm1_name))
    ppc_diagnostics_all[[as.character(i + 1)]] <- extract_posterior_predictive_data(
      fit = fit,
      y_true = y_test,
      cv_fold = i,
      prior_posterior_path = prior_posterior_path,
      n1 = n1,
      n2 = n / n1,
      y_transform = sampling_config$y_transform,
      adm1_code = df_test$adm1_code,
      adm1_name = df_test$adm1_name,
      datetime = as.Date(df_test$Datetime, tz = "GMT")
    )

    rMSE <- c(rMSE, sqrt(mean(ppc_diagnostics_all[[as.character(i + 1)]]$residual^2)))
    MAE <- c(MAE, mean(abs(ppc_diagnostics_all[[as.character(i + 1)]]$residual)))
  }

  # ========== FINAL SUMMARY ==========
  cat("\n\n=============================================\n")
  cat("         FINAL RESULTS SUMMARY\n")
  cat("=============================================\n\n")
  cat("Model Performance:\n")
  cat("  Mean rMSE:", round(mean(rMSE), 4), "\n")
  cat("  Mean MAE:", round(mean(MAE), 4), "\n\n")

  folds_completed <- as.numeric(names(convergence_results)) - 1
  final_summary <- data.frame(
    cv_fold = folds_completed,
    rMSE = rMSE,
    MAE = MAE,
    converged = sapply(convergence_results, function(x) x$converged),
    max_rhat = sapply(convergence_results, function(x) x$diagnostics$max_rhat)
  )
  write.csv(final_summary, file.path(output_path, "final_cv_summary.csv"), row.names = FALSE)

  if (length(selected_features) > 0) {
    all_beta_summary <- do.call(rbind, beta_diagnostics_all)
    write.csv(all_beta_summary,
      file.path(prior_posterior_path, "beta_summary_all_folds.csv"),
      row.names = FALSE
    )
  }

  all_ppc_summary <- do.call(rbind, ppc_diagnostics_all)
  write.csv(all_ppc_summary,
    file.path(prior_posterior_path, "ppc_summary_all_folds.csv"),
    row.names = FALSE
  )

  ppc_stats <- all_ppc_summary %>%
    group_by(cv_fold) %>%
    summarise(
      n_observations = n(),
      coverage_95 = mean(in_ci_95) * 100,
      mean_abs_residual = mean(abs(residual)),
      rmse = sqrt(mean(residual^2)),
      .groups = "drop"
    )
  write.csv(ppc_stats,
    file.path(prior_posterior_path, "ppc_statistics_by_fold.csv"),
    row.names = FALSE
  )
  cat("Posterior Predictive Check Statistics:\n")
  print(ppc_stats, n = Inf)
  cat("\nOutputs saved in", output_path, "\n")

  invisible(list(
    rMSE = rMSE,
    MAE = MAE,
    convergence_results = convergence_results,
    beta_diagnostics_all = beta_diagnostics_all,
    ppc_diagnostics_all = ppc_diagnostics_all,
    output_path = output_path
  ))
}
