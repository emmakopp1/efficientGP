# Function to create and save convergence plots
create_convergence_plots <- function(fit, cv_fold, diagnostics_path, selected_features = NULL, hyperprior = NULL) {
  # Extract parameter names
  parameters <- fit$summary() |>
    filter(!grepl("y_new", variable)) |>
    pull(variable)

  # Convert to draws for bayesplot
  draws <- fit$draws()

  # 1. Trace plots for main parameters
  p_trace <- mcmc_trace(draws,
    pars = parameters[!grepl("beta", parameters)],
    facet_args = list(ncol = 2)
  ) +
    ggtitle(paste("Trace Plots - CV Fold", cv_fold)) +
    theme_minimal()

  ggsave(file.path(diagnostics_path, paste0("trace_plot_cv", cv_fold, ".png")),
    p_trace,
    width = 12, height = 8, dpi = 300
  )

  # 1b. Trace plots for beta parameters
  if (!is.null(selected_features)) {
    tryCatch(
      {
        p_trace_beta <- mcmc_trace(draws,
          regex_pars = "beta",
          facet_args = list(ncol = 2)
        ) +
          ggtitle(paste("Trace Plots - Beta Parameters - CV Fold", cv_fold)) +
          theme_minimal()

        beta_params <- fit$summary() |>
          filter(grepl("^beta\\[", variable)) |>
          pull(variable)

        if (length(beta_params) == length(selected_features)) {
          new_labels <- paste0(selected_features, " (", beta_params, ")")
          names(new_labels) <- beta_params

          p_trace_beta <- p_trace_beta +
            facet_wrap(~parameter, ncol = 2, labeller = as_labeller(new_labels))
        }

        ggsave(file.path(diagnostics_path, paste0("trace_plot_beta_cv", cv_fold, ".png")),
          p_trace_beta,
          width = 12, height = 8, dpi = 300
        )
      },
      error = function(e) {
        cat("  Note: Could not create beta trace plots\n")
      }
    )
  }

  # 2. Rhat plot - Other parameters (excluding beta)
  rhat_summary <- summarise_draws(draws, "rhat") |>
    filter(!grepl("y_new", variable))

  # Rhat for non-beta parameters
  rhat_summary_other <- rhat_summary |>
    filter(!grepl("beta", variable))

  rhat_ratios_other <- setNames(rhat_summary_other$rhat, rhat_summary_other$variable)

  p_rhat_other <- mcmc_rhat(rhat = rhat_ratios_other) +
    ggtitle(paste("Rhat Values (Other Parameters) - CV Fold", cv_fold)) +
    geom_hline(yintercept = 1.05, linetype = "dashed", color = "red") +
    theme_minimal()

  ggsave(file.path(diagnostics_path, paste0("rhat_plot_cv", cv_fold, ".png")),
    p_rhat_other,
    width = 10, height = 6, dpi = 300
  )

  # 2b. Rhat plot - Beta parameters
  if (!is.null(selected_features)) {
    tryCatch(
      {
        rhat_summary_beta <- rhat_summary |>
          filter(grepl("beta", variable))

        if (nrow(rhat_summary_beta) > 0) {
          rhat_ratios_beta <- setNames(rhat_summary_beta$rhat, rhat_summary_beta$variable)

          p_rhat_beta <- mcmc_rhat(rhat = rhat_ratios_beta) +
            ggtitle(paste("Rhat Values (Beta Parameters) - CV Fold", cv_fold)) +
            geom_hline(yintercept = 1.05, linetype = "dashed", color = "red") +
            theme_minimal()

          beta_params <- rhat_summary_beta$variable

          if (length(beta_params) == length(selected_features)) {
            new_labels <- paste0(selected_features, " (", beta_params, ")")
            names(new_labels) <- beta_params

            p_rhat_beta <- p_rhat_beta +
              scale_y_discrete(labels = new_labels)
          }

          ggsave(file.path(diagnostics_path, paste0("rhat_plot_beta_cv", cv_fold, ".png")),
            p_rhat_beta,
            width = 10, height = 6, dpi = 300
          )
        }
      },
      error = function(e) {
        cat("  Note: Could not create beta Rhat plots\n")
      }
    )
  }

  # 3. ESS plots - Other parameters (excluding beta)
  ess_summary <- summarise_draws(draws, "ess_bulk") |>
    filter(!grepl("y_new", variable)) |>
    mutate(ratios = ess_bulk / (niterations(draws) * nchains(draws)))

  # ESS for non-beta parameters
  ess_summary_other <- ess_summary |>
    filter(!grepl("beta", variable))

  ess_ratios_other <- setNames(ess_summary_other$ratios, ess_summary_other$variable)

  p_ess_other <- mcmc_neff(ratio = ess_ratios_other) +
    ggtitle(paste("Effective Sample Size Ratio (Other Parameters) - CV Fold", cv_fold)) +
    theme_minimal()

  ggsave(file.path(diagnostics_path, paste0("ess_plot_cv", cv_fold, ".png")),
    p_ess_other,
    width = 10, height = 6, dpi = 300
  )

  # 3b. ESS plots - Beta parameters
  if (!is.null(selected_features)) {
    tryCatch(
      {
        ess_summary_beta <- ess_summary |>
          filter(grepl("beta", variable))

        if (nrow(ess_summary_beta) > 0) {
          ess_ratios_beta <- setNames(ess_summary_beta$ratios, ess_summary_beta$variable)

          p_ess_beta <- mcmc_neff(ratio = ess_ratios_beta) +
            ggtitle(paste("Effective Sample Size Ratio (Beta Parameters) - CV Fold", cv_fold)) +
            theme_minimal()

          beta_params <- ess_summary_beta$variable

          if (length(beta_params) == length(selected_features)) {
            new_labels <- paste0(selected_features, " (", beta_params, ")")
            names(new_labels) <- beta_params

            p_ess_beta <- p_ess_beta +
              scale_y_discrete(labels = new_labels)
          }

          ggsave(file.path(diagnostics_path, paste0("ess_plot_beta_cv", cv_fold, ".png")),
            p_ess_beta,
            width = 10, height = 6, dpi = 300
          )
        }
      },
      error = function(e) {
        cat("  Note: Could not create beta ESS plots\n")
      }
    )
  }

  # 4. Autocorrelation plot - Other parameters (excluding beta)
  params_other <- parameters[!grepl("beta", parameters)]

  p_acf_other <- mcmc_acf(draws,
    pars = params_other,
    lags = 20
  ) +
    ggtitle(paste("Autocorrelation (Other Parameters) - CV Fold", cv_fold)) +
    theme_minimal()

  ggsave(file.path(diagnostics_path, paste0("acf_plot_cv", cv_fold, ".png")),
    p_acf_other,
    width = 16, height = 8, dpi = 300
  )

  # 4b. Autocorrelation plot - Beta parameters
  if (!is.null(selected_features)) {
    tryCatch(
      {
        params_beta <- parameters[grepl("beta", parameters)]

        if (length(params_beta) > 0) {
          beta_params <- parameters[grepl("^beta\\[", parameters)]

          if (length(beta_params) == length(selected_features)) {
            new_labels <- paste0(selected_features)
            names(new_labels) <- beta_params

            p_acf_beta <- mcmc_acf(draws,
              pars = params_beta,
              lags = 20,
              facet_args = list(labeller = as_labeller(new_labels))
            ) +
              ggtitle(paste("Autocorrelation (Beta Parameters) - CV Fold", cv_fold)) +
              theme_minimal() +
              theme(strip.text = element_text(size = 9))
          } else {
            p_acf_beta <- mcmc_acf(draws,
              pars = params_beta,
              lags = 20
            ) +
              ggtitle(paste("Autocorrelation (Beta Parameters) - CV Fold", cv_fold)) +
              theme_minimal()
          }

          ggsave(file.path(diagnostics_path, paste0("acf_plot_beta_cv", cv_fold, ".png")),
            p_acf_beta,
            width = 16, height = 8, dpi = 300
          )
        }
      },
      error = function(e) {
        cat("  Note: Could not create beta ACF plots\n")
      }
    )
  }

  # 5. Areas plot for beta parameters
  if (!is.null(selected_features)) {
    # Extract beta parameters
    # Calculer quels paramètres incluent 0
    beta_summary <- fit$summary(
      variables = NULL,
      quantiles = ~ quantile(.x, probs = c(0, 0.025, 0.5, 0.975, 1))
    ) |>
      filter(grepl("^beta\\[", variable)) |>
      mutate(
        includes_zero = `2.5%` <= 0 & `97.5%` >= 0,
        color = ifelse(includes_zero, "darkred", "darkblue")
      )

    # Ajouter les noms de features si disponibles
    if (length(selected_features) == nrow(beta_summary)) {
      beta_summary$feature_label <- paste0(selected_features, " (", beta_summary$variable, ")")
    } else {
      beta_summary$feature_label <- beta_summary$variable
    }

    # Créer le graphique
    p_intervals <- beta_summary |>
      ggplot(aes(
        y = reorder(feature_label, `50%`), x = `50%`,
        color = color, fill = color
      )) +
      # Ligne fine pour l'étendue totale (0% à 100%)
      geom_segment(aes(x = `0%`, xend = `100%`, yend = feature_label),
        linewidth = 0.8, alpha = 0.4
      ) +
      # Ligne épaisse pour l'IC 95% (2.5% à 97.5%)
      geom_segment(aes(x = `2.5%`, xend = `97.5%`, yend = feature_label),
        linewidth = 3
      ) +
      # Point pour la médiane
      geom_point(size = 4, color = "white", shape = 21, stroke = 1.5) +
      # Ligne verticale à 0
      geom_vline(
        xintercept = 0, linetype = "dashed", color = "gray40",
        linewidth = 0.5, alpha = 0.7
      ) +
      scale_color_identity() +
      scale_fill_identity() +
      labs(x = "Posterior", y = NULL) +
      theme_minimal(base_size = 12) +
      theme(
        axis.text.y = element_text(hjust = 1, size = 11),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank()
      )
    ggsave(file.path(diagnostics_path, paste0("interval_plot_beta_cv", cv_fold, ".png")),
      p_intervals,
      width = 12, height = 10, dpi = 300
    )
  }


  # 6. Areas plot for hyperprior parameter(s)
  if (!is.null(hyperprior)) {
    for (hp in hyperprior) {
      tryCatch(
        {
          if (hp %in% parameters) {
            color_scheme_set("purple")
            p_areas_hyperprior <- mcmc_areas(draws,
              pars = hp,
              prob = 0.95, area_method = "scaled height"
            ) +
              ggtitle(paste(
                "Posterior Distribution of", hp,
                "- CV Fold", cv_fold
              )) +
              theme_minimal()

            ggsave(
              file.path(
                diagnostics_path,
                paste0("areas_plot_", hp, "_cv", cv_fold, ".png")
              ),
              p_areas_hyperprior,
              width = 10, height = 5, dpi = 300
            )
            cat("  ✓", hp, "area plot saved\n")
          } else {
            cat("  Note:", hp, "not found in model parameters\n")
          }
        },
        error = function(e) {
          cat("  Note: Could not create", hp, "area plot\n")
        }
      )
    }
  }

  cat("  ✓ Convergence plots saved to:", diagnostics_path, "\n")
}

# Function to extract and save data for beta prior-posterior analysis
extract_beta_diagnostics <- function(fit, cv_fold, prior_posterior_path, selected_features) {
  if (is.null(selected_features)) {
    return()
  }

  cat("\n--- Extracting Beta Diagnostics for CV Fold", cv_fold, "---\n")

  # Extract beta samples
  beta_draws <- fit$draws("beta", format = "matrix")
  n_samples <- nrow(beta_draws)
  P <- ncol(beta_draws)

  # VALIDATION: Check that P > 0
  if (is.null(P) || P == 0) {
    cat("  ⚠️  Warning: No beta parameters found. Skipping beta diagnostics.\n")
    return(NULL)
  }


  # Create summary table for each beta
  beta_summary <- data.frame(
    cv_fold = cv_fold,
    feature = selected_features,
    beta_name = paste0("beta[", 1:P, "]"),
    posterior_mean = colMeans(beta_draws),
    posterior_sd = apply(beta_draws, 2, sd),
    posterior_median = apply(beta_draws, 2, median),
    ci_2.5 = apply(beta_draws, 2, quantile, probs = 0.025),
    ci_97.5 = apply(beta_draws, 2, quantile, probs = 0.975),
    prob_positive = colMeans(beta_draws > 0),
    prob_negative = colMeans(beta_draws < 0),
    includes_zero = apply(beta_draws, 2, function(x) {
      ci <- quantile(x, probs = c(0.025, 0.975))
      ci[1] <= 0 & ci[2] >= 0
    }),
    prior_mean = 0 # (ridge and normal prior is always centered at 0
  )

  # Add ESS and Rhat for each beta
  draws_obj <- fit$draws("beta")
  beta_diagnostics <- summarise_draws(draws_obj, "rhat", "ess_bulk", "ess_tail")
  beta_summary$rhat <- beta_diagnostics$rhat
  beta_summary$ess_bulk <- beta_diagnostics$ess_bulk
  beta_summary$ess_tail <- beta_diagnostics$ess_tail

  # Save summary table
  write.csv(beta_summary,
    file.path(prior_posterior_path, paste0("beta_summary_cv", cv_fold, ".csv")),
    row.names = FALSE
  )

  # Save full posterior samples for each beta (for plotting later)
  beta_samples_df <- as.data.frame(beta_draws)
  colnames(beta_samples_df) <- paste0("beta[", 1:P, "]")
  beta_samples_df$iteration <- 1:n_samples
  beta_samples_df$cv_fold <- cv_fold
  write.csv(beta_samples_df,
    file.path(prior_posterior_path, paste0("beta_posterior_samples_cv", cv_fold, ".csv")),
    row.names = FALSE
  )

  cat("  ✓ Beta diagnostics saved:", nrow(beta_summary), "coefficients\n")

  return(beta_summary)
}

# Function to extract and save data for posterior predictive check
extract_posterior_predictive_data <- function(fit, y_true, cv_fold, prior_posterior_path, n1, n2, y_transform = TRUE,
                                              adm1_code = NULL, adm1_name = NULL, datetime = NULL, write_res = TRUE,
                                              cv_fold_nested = NULL, rho1_cv = NULL) {
  if (length(cv_fold_nested) > 0) {
    cat("\n--- Extracting Posterior Predictive Data for CV Fold", cv_fold, "Nested CV Fold", cv_fold_nested, "and rho1", rho1_cv, "---\n")
  } else if (length(cv_fold_nested) == 0) {
    cat("\n--- Extracting Posterior Predictive Data for CV Fold", cv_fold, "---\n")
  }

  # Extract y_new samples (predictions) in logit space
  y_new_draws_logit <- fit$draws("y_new", format = "matrix")
  n_samples <- nrow(y_new_draws_logit)
  n_pred <- n1 * n2

  # Apply inverse logit transformation to convert back to original scale [0, 1]
  if (y_transform == TRUE) {
    y_new_draws <- suppressWarnings(1 / (1 + exp(-y_new_draws_logit[, 1:n_pred])))
  } else if (y_transform == FALSE) {
    y_new_draws <- y_new_draws_logit
  }

  # Calculate summary statistics for predictions (now in original scale)
  y_pred_mean <- colMeans(y_new_draws)
  y_pred_sd <- apply(y_new_draws, 2, sd)
  y_pred_quantiles <- t(apply(y_new_draws, 2, quantile, probs = c(0.025, 0.25, 0.5, 0.75, 0.975)))

  # Create summary dataframe
  ppc_summary <- data.frame(
    cv_fold = cv_fold,
    observation_id = 1:n_pred,
    y_true = y_true,
    y_pred_mean = y_pred_mean,
    y_pred_sd = y_pred_sd,
    y_pred_median = y_pred_quantiles[, 3],
    y_pred_ci_2.5 = y_pred_quantiles[, 1],
    y_pred_ci_97.5 = y_pred_quantiles[, 5],
    y_pred_q25 = y_pred_quantiles[, 2],
    y_pred_q75 = y_pred_quantiles[, 4],
    residual = y_pred_mean - y_true,
    in_ci_95 = y_true >= y_pred_quantiles[, 1] & y_true <= y_pred_quantiles[, 5],
    len_ic_95 = y_pred_quantiles[, 5] - y_pred_quantiles[, 1]
  )

  # Add spatial and temporal information if provided
  if (!is.null(adm1_code)) {
    ppc_summary$adm1_code <- adm1_code
  }
  if (!is.null(adm1_name)) {
    ppc_summary$adm1_name <- adm1_name
  }
  if (!is.null(datetime)) {
    ppc_summary$Datetime <- datetime
  }

  # Save summary
  if (write_res) {
    write.csv(ppc_summary,
      file.path(prior_posterior_path, paste0("ppc_summary_cv", cv_fold, ".csv")),
      row.names = FALSE
    )
  }

  # Save a subset of posterior predictive samples (e.g., 100 random draws for visualization)
  n_samples_to_save <- min(100, n_samples)
  sample_indices <- sample(1:n_samples, n_samples_to_save)
  y_new_subset <- y_new_draws[sample_indices, ]

  ppc_samples_df <- as.data.frame(t(y_new_subset))
  colnames(ppc_samples_df) <- paste0("sample_", 1:n_samples_to_save)
  ppc_samples_df$observation_id <- 1:n_pred
  ppc_samples_df$y_true <- y_true
  ppc_samples_df$cv_fold <- cv_fold

  if (write_res) {
    write.csv(ppc_samples_df,
      file.path(prior_posterior_path, paste0("ppc_samples_cv", cv_fold, ".csv")),
      row.names = FALSE
    )
  }

  # Calculate coverage statistics
  coverage_95 <- mean(ppc_summary$in_ci_95)

  cat("  ✓ Posterior predictive data saved\n")
  cat("    - Number of predictions:", n_pred, "\n")
  cat("    - 95% CI coverage:", round(coverage_95 * 100, 2), "%\n")
  cat("    - Average 95% CI length:", round(mean(ppc_summary$len_ic_95), 2), "\n")
  cat("    - Mean absolute residual:", round(mean(abs(ppc_summary$residual)), 4), "\n")

  return(ppc_summary)
}


# Function to extract and save posterior mean of parameters
extract_posterior_mean <- function(fit, cv_fold, output_path) {
  # Extract parameter summaries
  params <- fit$summary() |>
    filter(!grepl("y_new", variable)) |>
    pull(variable)

  cat("\n--- Extracting Posterior Mean for CV Fold", cv_fold, "---\n")

  # Extract posterior mean
  post_mean <- fit$summary(variables = params)$mean
  names(post_mean) <- params

  # Save to CSV
  write.csv(post_mean,
    file.path(output_path, paste0("post_mean_samples_cv", cv_fold, ".csv")),
    row.names = TRUE
  )

  cat("  ✓ Posterior mean saved:", length(post_mean), "parameters\n")

  return(post_mean)
}

# Function to extract and save posterior mean of parameters
extract_posterior_mean_llkd <- function(fit, cv_fold, output_path) {
  cat("\n--- Extracting Posterior Mean of the log-likelihood for CV Fold", cv_fold, "---\n")

  # Extract posterior mean
  post_mean_llkd <- fit$summary(variables = "lp__")$mean
  names(post_mean_llkd) <- "llkd"

  # Save to CSV
  write.csv(post_mean_llkd,
    file.path(output_path, paste0("post_mean_samples_llkd_cv", cv_fold, ".csv")),
    row.names = TRUE
  )

  cat("  ✓ Log-Likelihood mean saved:", length(post_mean_llkd), "parameters\n")

  return(post_mean_llkd)
}
