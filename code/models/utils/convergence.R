# Function to check convergence and recommend iterations
check_convergence <- function(fit, cv_fold, diagnostics_path, write_res = TRUE, cv_fold_nested = NULL, rho1_cv = NULL) {
  # Extract diagnostics
  diagnostics <- fit$diagnostic_summary()

  # Convert to draws object for posterior functions
  draws <- fit$draws()
  rhat_values <- summarise_draws(draws, "rhat")$rhat
  ess_bulk_values <- summarise_draws(draws, "ess_bulk")$ess_bulk
  ess_tail_values <- summarise_draws(draws, "ess_tail")$ess_tail

  # Print summary
  if (length(cv_fold_nested) > 0) {
    cat("\n", strrep("=", 50), "\n", sep = "")
    cat("├─ CV Fold:", cv_fold, "\n")
    cat("├─ Nested Fold:", cv_fold_nested, "\n")
    cat("├─ Rho1:", rho1_cv, "\n")
    cat("└─ Convergence Summary:\n")
    cat(strrep("=", 50), "\n\n", sep = "")
  } else if (length(cv_fold_nested) == 0) {
    cat("\n========== CV Fold", cv_fold, "Convergence Summary ==========\n")
  }
  cat("Max Rhat:", max(rhat_values, na.rm = TRUE), "\n")
  cat("Min ESS Bulk:", min(ess_bulk_values, na.rm = TRUE), "\n")
  cat("Min ESS Tail:", min(ess_tail_values, na.rm = TRUE), "\n")
  cat("Divergent transitions:", sum(diagnostics$num_divergent), "\n")
  cat("Max treedepth hits:", sum(diagnostics$num_max_treedepth), "\n")

  # Save numerical diagnostics
  convergence_summary <- data.frame(
    cv_fold = cv_fold,
    max_rhat = max(rhat_values, na.rm = TRUE),
    min_ess_bulk = min(ess_bulk_values, na.rm = TRUE),
    min_ess_tail = min(ess_tail_values, na.rm = TRUE),
    num_divergent = sum(diagnostics$num_divergent),
    num_max_treedepth = sum(diagnostics$num_max_treedepth)
  )

  if (write_res) {
    write.csv(convergence_summary,
      file.path(diagnostics_path, paste0("convergence_summary_cv", cv_fold, ".csv")),
      row.names = FALSE
    )
  }

  # Determine if converged
  converged <- max(rhat_values, na.rm = TRUE) < 1.05 &&
    min(ess_bulk_values, na.rm = TRUE) > 400 &&
    sum(diagnostics$num_divergent) == 0

  # Recommendations
  if (!converged) {
    cat("\n⚠️  CONVERGENCE ISSUES DETECTED!\n")
    if (max(rhat_values, na.rm = TRUE) >= 1.05) {
      cat("  - Rhat > 1.05: Consider increasing iterations\n")
    }
    if (min(ess_bulk_values, na.rm = TRUE) < 400) {
      cat("  - Low ESS: Consider increasing iterations or thinning\n")
    }
    if (sum(diagnostics$num_divergent) > 0) {
      cat("  - Divergent transitions: Consider increasing adapt_delta\n")
    }
  } else {
    cat("\n✓ Convergence looks good!\n")
  }

  return(list(converged = converged, diagnostics = convergence_summary))
}


# Adaptive sampling function
adaptive_sample <- function(mod, data_list, seed, max_attempts = 3, warmup_init = 400, sampling_init = 700, adapt_delta_init = 0.8, max_treedepth_init = 10, n_chains = 4, n_parallel_chains = 4) {
  # Initial sampling parameters
  warmup <- warmup_init
  sampling <- sampling_init
  adapt_delta <- adapt_delta_init
  max_treedepth <- max_treedepth_init

  for (attempt in 1:max_attempts) {
    cat("\n--- Sampling Attempt", attempt, "---\n")
    cat(
      "Warmup:", warmup, "| Sampling:", sampling,
      "| adapt_delta:", adapt_delta, "| max_treedepth:", max_treedepth, "\n"
    )

    fit <- mod$sample(
      data = data_list,
      seed = seed,
      iter_warmup = warmup,
      iter_sampling = sampling,
      save_warmup = TRUE,
      chains = n_chains,
      parallel_chains = n_parallel_chains,
      refresh = 50,
      adapt_delta = adapt_delta,
      max_treedepth = max_treedepth
    )

    # Check convergence using posterior package
    draws <- fit$draws()
    rhat_summary <- summarise_draws(draws, "rhat")
    ess_bulk_summary <- summarise_draws(draws, "ess_bulk")

    rhat_max <- max(rhat_summary$rhat, na.rm = TRUE)
    ess_min <- min(ess_bulk_summary$ess_bulk, na.rm = TRUE)
    diagnostics <- fit$diagnostic_summary()
    num_divergent <- sum(diagnostics$num_divergent)

    # If converged, return
    if (rhat_max < 1.05 && ess_min > 400 && num_divergent == 0) {
      cat("✓ Convergence achieved!\n")
      return(fit)
    }

    # Adjust parameters for next attempt
    if (rhat_max >= 1.05 || ess_min < 400) {
      warmup <- warmup + 200
      sampling <- sampling + 300
    }
    if (num_divergent > 0) {
      adapt_delta <- min(0.99, adapt_delta + 0.05)
    }
    if (sum(diagnostics$num_max_treedepth) > 0) {
      max_treedepth <- min(15, max_treedepth + 2)
    }
  }

  cat("⚠️  Warning: Did not achieve full convergence after", max_attempts, "attempts\n")
  return(fit)
}
