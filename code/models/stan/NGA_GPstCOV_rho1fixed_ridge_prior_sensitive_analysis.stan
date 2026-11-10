#include GP_helper.stan
// Sensitivity model for dependence induced by overlapping rolling survey windows.
// Same mean and additive GP structure as GPstCOV_rho1fixed_ridge_centered_prior.stan, plus
// a region-specific overlap error e_r ~ N(0, tau_ov^2 * R_ov), independent across regions:
//   K_y = alpha0^2 * (J_S + alpha1^2 K1c) (x) (J_T + K2c) + sigma1^2 * I_S (x) J_T
//         + tau_ov^2 * I_S (x) R_ov + sigma^2 * I
// with K2c = alpha21^2 K21c + alpha22^2 K22c and
//   (R_ov)_{tt'} = max(0, 1 - |d_t - d_t'| / window_days).
// All spatial factors share the fixed eigenbasis Q1, so rotating the residuals by Q1
// splits the likelihood into N1 independent N2-dimensional Gaussians with covariance
//   A_j = alpha0^2 * lambda_j * (J_T + K2c) + sigma1^2 J_T + tau_ov^2 R_ov + sigma^2 I,
// where lambda_1 = N1 (constant direction) and lambda_j = alpha1^2 * l1[j] otherwise.
// Requires a complete region-by-time grid (as the reference model does).
//
// generated quantities: overall posterior of the scientific latent process
// (mean + additive GP + spatial random effect), matching y_new in the reference
// model. Overlap and nugget enter the training covariance used for conditioning
// but are not added to the predictive covariance of y_new.
data {
  int<lower=1> N1 ; // dimension espace
  int<lower=1> N2 ; // dimension temps
  int<lower=1> n1 ;// number of new locations in space
  int<lower=2> n2 ; // number of new locations in time
  int<lower=1> P; // number of covariates
  matrix[N1,2] X1; // training spatial locations
  matrix[N2,1] X2; // training time locations
  matrix[N1*N2, P] D; // covariate matrix
  vector[N1*N2] y ; // training observations (region-major, time within region)
  matrix[n1, 2] x1_new ; // test spatial locations
  matrix[n2, 1] x2_new ; // test time locations
  matrix[n1*n2, P] D_new; // covariate matrix for test
  real<lower=0> rho1 ; // range parameter for spatial kernel (fixed)
  real<lower=0,upper=1> Hurst2; // Hurst parameter for fBM kernel (fixed)
  vector[N2] t_days; // observation dates in days (common to all regions), same order as X2
}
transformed data {
  int N = N1*N2 ;
  int n = n1*n2;
  real window_days = 90; // rolling FCS window width
  real tau_ov_scale = 0.5; // half-normal scale for tau_ov on the logit scale
  real jitter = 1e-8; // numerical jitter, kept separate from sigma
  vector[N1] l1; // eigenvalues for spatial kernel (after centering)
  matrix[N1, N1] Q1 ; // eigenvectors for spatial kernel (after centering)
  vector[N1] lambda1; // eigenvalues of J_S in the Q1 basis
  vector[N1] K1rowsum;
  vector[N2] K22rowsum;
  matrix[N2, N2] K22; // covariance matrix for temporal fBM kernel (centred)
  matrix[N2, N2] J2 = rep_matrix(1, N2, N2);
  matrix[N2, N2] R_ov; // overlap correlation (triangular kernel, PSD in 1D)
  matrix[N2, N1] Z; // spatially rotated observations
  {
    matrix[N1, N1] K1 = Gram_exponential(X1,rho1);
    K1rowsum = K1 * rep_vector(1,N1);
    K1 = Gram_centring(K1);
    {
      matrix[N1, N1+1] R = cen_eigen_decompose(K1);
      Q1 = R[1:N1, 1:N1] ; // eigenvectors, Q1[,1] = constant direction
      l1 = to_vector(R[1:N1, N1+1]); // eigenvalues, l1[1] = 0
    }
    K22 = Gram_fBM(X2, Hurst2);
    K22rowsum = K22 * rep_vector(1,N2);
    K22 = Gram_centring(K22);
  }
  lambda1 = rep_vector(0, N1);
  lambda1[1] = N1;
  for (i in 1:N2) {
    R_ov[i,i] = 1;
    for (j in 1:(i-1)) {
      R_ov[i,j] = fmax(0, 1 - abs(t_days[i] - t_days[j]) / window_days);
      R_ov[j,i] = R_ov[i,j];
    }
  }
  if (min(eigenvalues_sym(R_ov)) < -1e-8)
    reject("R_ov is not positive semi-definite; check t_days");
  Z = to_matrix(y, N2, N1) * Q1;
}
// Paramètres à estimer
parameters {
  real<lower=0> alpha0 ;
  real<lower=0> alpha1 ;//scale parameter for spatial kernel
  real<lower=0> alpha21; //scale parameter for exponential kernel (time)
  real<lower=0> alpha22; //scale parameter for fBM kernel (time)
  real<lower=0, upper=N2> rho2;
  real<lower=0> sigma1 ; // random effect for space
  real<lower=0> sigma ;
  vector[P] beta ; // regression coefficients
  real<lower=0> kappa ; // global shrinkage parameter for ridge prior
  real<lower=0> tau_ov ; // marginal sd of the overlap error
}
model{ // prior + likelihood
  //prior
  target += lognormal_lpdf(alpha0|0,1);
  target += lognormal_lpdf(alpha1|0,1);
  target += lognormal_lpdf(alpha21|0,1);
  target += lognormal_lpdf(alpha22|0,1);
  target += inv_gamma_lpdf(rho2|2,5);
  target += lognormal_lpdf(sigma1|0,1);
  target += lognormal_lpdf(sigma|0,1);
  target += normal_lpdf(tau_ov|0, tau_ov_scale); // half-normal, allows tau_ov -> 0

  // Ridge Bayesian prior on beta
  kappa ~ gamma(0.5, 0.5);
  beta ~ multi_normal(rep_vector(0, P), (1/kappa) * identity_matrix(P));

  //likelihood
  // Z[,j] ~ N(H[,j], A_j), j = 1..N1, with Z = Y Q1 and H = M Q1
  {
    matrix[N2, N1] H = to_matrix(D*beta, N2, N1) * Q1; // rotated mean
    matrix[N2, N2] K2; // J_T + K2c
    matrix[N2, N2] B; // block shared by all spatial directions
    {
      matrix[N2,N2] K21 = Gram_exponential(X2, rho2);
      K21 = Gram_centring(K21);
      K2 = J2 + square(alpha21)*K21 + square(alpha22)*K22;
    }
    B = add_diag(square(sigma1)*J2 + square(tau_ov)*R_ov, square(sigma) + jitter);
    for (j in 1:N1) {
      real lambda_j = lambda1[j] + square(alpha1) * l1[j];
      matrix[N2, N2] L_j = cholesky_decompose(square(alpha0) * lambda_j * K2 + B);
      target += multi_normal_cholesky_lpdf(Z[,j] | H[,j], L_j);
    }
  }
}
generated quantities {
  vector[n] y_new;
  {
    vector[n] mu_new;
    matrix[n, n] L_new;
    {
      matrix[N2, N1] H = to_matrix(D*beta, N2, N1) * Q1;
      matrix[N2, N2] K2;
      matrix[N2, N2] Bblk;
      vector[N2] K21rowsum;
      matrix[n1, n1] K1_new;
      matrix[n2, n2] K2_new;
      matrix[N1, n1] K1_trnew;
      matrix[N2, n2] K2_trnew;
      matrix[N1, n1] RE_trnew;
      {
        matrix[N2,N2] K21 = Gram_exponential(X2, rho2);
        K21rowsum = K21 * rep_vector(1,N2);
        K21 = Gram_centring(K21);
        K2 = J2 + square(alpha21)*K21 + square(alpha22)*K22;
      }
      Bblk = add_diag(square(sigma1)*J2 + square(tau_ov)*R_ov, square(sigma) + jitter);
      {
        vector[n1] k1tmp;
        vector[n2] k21tmp;
        vector[n2] k22tmp;
        matrix[N2, n2] K21_trnew;
        matrix[N2, n2] K22_trnew;
        for (i in 1:n1){
          K1_trnew[,i] = kvec_exponential(X1, to_vector(x1_new[i,]), rho1);
          k1tmp[i] = sum(K1_trnew[,i]);
          for (j in 1:i){
            K1_new[i,j] =  1 + square(alpha1)*(kernel_exponential(to_vector(x1_new[i,]),to_vector(x1_new[j,]),rho1) - k1tmp[i]/N1 - k1tmp[j]/N1 +sum(K1rowsum)/square(N1));
            K1_new[j,i] = K1_new[i,j];
          }
        }
        for (i in 1:n1){
          K1_trnew[,i] = rep_vector(1,N1) + square(alpha1)*kvec_cen(K1_trnew[,i] ,K1rowsum);
          for (j in 1:N1){
            if (distance(x1_new[i,], X1[j,]) < 1e-6) {
              RE_trnew[j,i] = 1 ;
            } else {
              RE_trnew[j,i] = 0 ;
            }
          }
        }
        for (i in 1:n2){
          K21_trnew[,i] = kvec_exponential(X2, to_vector(x2_new[i,]), rho2);
          K22_trnew[,i] = kvec_fBM(X2, to_vector(x2_new[i,]), Hurst2);
          k21tmp[i] = sum(K21_trnew[,i]);
          k22tmp[i] = sum(K22_trnew[,i]);
          for (j in 1:i){
            real k21star = kernel_exponential(to_vector(x2_new[i,]),to_vector(x2_new[j,]),rho2) - k21tmp[i]/N2 - k21tmp[j]/N2 +sum(K21rowsum)/square(N2);
            real k22star = kernel_fBM(to_vector(x2_new[i,]),to_vector(x2_new[j,]),Hurst2) - k22tmp[i]/N2 - k22tmp[j]/N2 +sum(K22rowsum)/square(N2);
            K2_new[i,j] =  1 + square(alpha21)* k21star + square(alpha22)* k22star;
            K2_new[j,i] = K2_new[i,j];
          }
        }
        for (i in 1:n2){
          K2_trnew[,i] = rep_vector(1,N2) + square(alpha21)*kvec_cen(K21_trnew[,i] ,K21rowsum) + square(alpha22)*kvec_cen(K22_trnew[,i] ,K22rowsum);
        }
      }
      {
        matrix[N1, n1] G1 = Q1' * K1_trnew;
        matrix[N1, n1] C1 = Q1' * RE_trnew;
        matrix[n, n] K_new = square(alpha0)*kronecker_prod(K1_new,K2_new)
          + square(sigma1)*kronecker_prod(diag_matrix(rep_vector(1, n1)),rep_matrix(1,n2,n2));
        mu_new = D_new * beta;
        for (j in 1:N1) {
          real lambda_j = lambda1[j] + square(alpha1) * l1[j];
          matrix[N2, N2] L_j = cholesky_decompose(square(alpha0) * lambda_j * K2 + Bblk);
          matrix[N2, n] M_j = square(alpha0) * kronecker_prod(to_matrix(G1[j]), K2_trnew)
            + square(sigma1) * kronecker_prod(to_matrix(C1[j]), rep_matrix(1, N2, n2));
          matrix[N2, n] V_j = mdivide_left_tri_low(L_j, M_j);
          vector[N2] u_j = mdivide_left_tri_low(L_j, Z[,j] - H[,j]);
          mu_new += V_j' * u_j;
          K_new -= V_j' * V_j;
        }
        L_new = cholesky_decompose(K_new + diag_matrix(rep_vector(1e-9,n)));
      }
    }
    y_new = multi_normal_cholesky_rng(mu_new, L_new);
  }
}
