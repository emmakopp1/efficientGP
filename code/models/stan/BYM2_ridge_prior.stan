/*
  Spatio-temporal BYM2 benchmark on logit(FCS), with a ridge prior on the
  regression coefficients.

    y_it      ~ normal(eta_it, sigma)                 (y = logit FCS prevalence)
    eta_it    = b0 + D_it * beta + s_i + nu_t + xi_t
    s_i       = sigma_s * ( sqrt(rho / scaling_factor) * phi_i + sqrt(1 - rho) * theta_i )
    phi       ~ ICAR(graph), soft sum-to-zero          (structured spatial)
    theta_i   ~ normal(0, 1)                           (unstructured spatial)
    nu_t - nu_{t-1} ~ normal(0, sigma_nu), soft sum-to-zero (structured temporal)
    xi_t      ~ normal(0, sigma_xi)                    (unstructured temporal)

  No space-time interaction term.

  Regions are nodes of the adjacency graph. All nodes are parameters, but only
  the training observations enter the likelihood. For a held-out region, phi_i
  is therefore informed by its neighbours through the ICAR prior, while theta_i
  is drawn from its prior. Predictions are produced in generated quantities.

  References:
    Riebler et al. (2016), Stat. Methods Med. Res. 25(4):1145-1165
    Morris et al. (2019), Spat. Spatiotemporal Epidemiol. 31:100301
    https://mc-stan.org/learn-stan/case-studies/icar_stan.html
    https://github.com/MLGlobalHealth/Realtime-SAE-Zimbabwe
*/
functions {
  // ICAR prior with a soft sum-to-zero constraint
  real icar_normal_lpdf(vector phi, int N, array[] int node1, array[] int node2) {
    return -0.5 * dot_self(phi[node1] - phi[node2])
           + normal_lpdf(sum(phi) | 0, 0.001 * N);
  }
}
data {
  int<lower=1> NS; // number of regions (nodes of the graph)
  int<lower=2> NT; // number of time points (weeks)
  int<lower=0> P; // number of covariates

  // training data
  int<lower=1> N;
  vector[N] y; // logit(FCS)
  matrix[N, P] D; // standardised covariates
  array[N] int<lower=1, upper=NS> area; // region index of each observation
  array[N] int<lower=1, upper=NT> time; // time index of each observation

  // prediction data
  int<lower=1> n;
  matrix[n, P] D_new;
  array[n] int<lower=1, upper=NS> area_new;
  array[n] int<lower=1, upper=NT> time_new;

  // adjacency graph
  int<lower=1> N_edges;
  array[N_edges] int<lower=1, upper=NS> node1;
  array[N_edges] int<lower=1, upper=NS> node2;
  real<lower=0> scaling_factor; // geometric mean of the ICAR marginal variances

  // PC prior on standard deviations: P(sd > pc_u) = pc_alpha
  real<lower=0> pc_u;
  real<lower=0, upper=1> pc_alpha;
}
transformed data {
  real lambda_pc = -log(pc_alpha) / pc_u;
}
parameters {
  real b0; // intercept
  vector[P] beta; // regression coefficients
  real<lower=0> kappa; // ridge precision

  vector[NS] phi; // ICAR component (unit scale)
  vector[NS] theta; // iid spatial component (unit scale)
  real<lower=0> sigma_s; // marginal sd of the spatial effect
  real<lower=0, upper=1> rho; // proportion of spatial variance that is structured

  vector[NT] nu; // RW1 (centred: the weekly level is well identified by the data)
  vector[NT] u_xi; // iid temporal (unit scale)
  real<lower=0> sigma_nu;
  real<lower=0> sigma_xi;

  real<lower=0> sigma; // observation noise
}
transformed parameters {
  vector[NS] s = sigma_s * (sqrt(rho / scaling_factor) * phi + sqrt(1 - rho) * theta);
  vector[NT] xi = sigma_xi * u_xi;
}
model {
  // likelihood
  target += normal_lpdf(y | b0 + D * beta + s[area] + nu[time] + xi[time], sigma);

  // spatial effects
  target += icar_normal_lpdf(phi | NS, node1, node2);
  target += std_normal_lpdf(theta);

  // temporal effects
  target += normal_lpdf(nu[2:NT] | nu[1:(NT - 1)], sigma_nu);
  target += normal_lpdf(sum(nu) | 0, 0.001 * NT);
  target += std_normal_lpdf(u_xi);

  // fixed effects: ridge prior, same as GPstCOV_rho1fixed_ridge_centered_prior.stan
  target += normal_lpdf(b0 | 0, 5);
  target += gamma_lpdf(kappa | 0.5, 0.5);
  target += normal_lpdf(beta | 0, inv_sqrt(kappa));

  // hyperpriors: PC priors on the standard deviations (exponential),
  // equivalent to the pcprec prior on the precisions of the template
  target += exponential_lpdf(sigma_s | lambda_pc);
  target += exponential_lpdf(sigma_nu | lambda_pc);
  target += exponential_lpdf(sigma_xi | lambda_pc);
  target += exponential_lpdf(sigma | lambda_pc);
  target += beta_lpdf(rho | 0.5, 0.5);
}
generated quantities {
  vector[n] y_new; // posterior predictive draws for the held-out observations (logit scale)
  {
    vector[n] mu_new = b0 + D_new * beta + s[area_new] + nu[time_new] + xi[time_new];
    for (k in 1:n) {
      y_new[k] = normal_rng(mu_new[k], sigma);
    }
  }
}
