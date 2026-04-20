/* functions related to eigendecomposition of centered Gram matrix */
functions {
int n_zero_eval(vector eval){
    // count number of zero eigen values (smaller than threshold)
  // Input: eval - a vector of eigenvalues
  //        n ---- a size of a matrix
  // Output: number of zero eigen values
  int n = num_elements(eval);
  int k = 0;
  for (i in 1:n){
    if (eval[i] < 1e-9) k += 1;
    else break;
  }
  return k;
}

vector eval_zero(vector eval, int k){
  // replace eigenvalues with zero
  int n = num_elements(eval);
  vector[n] evalz = eval;
  for (i in 1:k)  evalz[i] = 0.0;
  return evalz ;
}

matrix GS_complete(matrix Evec, int k){
  // Gram-Schmidt process
  // Input: Evec - original sets of eigenvaectors
  //        k -the number of zero eigen values
  // Output: sets of eigenvaectors after Gram-Schumidt process
  matrix[rows(Evec),k] Q;
  matrix[rows(Evec),k] V ;
  int n = rows(Evec);
  matrix[n,k] X = Evec[,1:k];
  X[,1] =  rep_vector(1/sqrt(n),n) ;
  V = X ;
  Q[,1] =  V[,1];
  for (i in 2:k){
    for (j in 1:(i-1)){
      V[,i] = V[,i] - ((Q[,j]')*X[,i])*Q[,j];
    }
    Q[,i] = V[,i]/sqrt(sum(V[,i].*V[,i]));
  }
  return append_col(Q,Evec[,(k+1):n]);
}

matrix cen_eigen_decompose(matrix K){
  // Output: Eigenvalues and eigenvector of a centered Gram matrix
  // Input: K - Centred Gram matrix
  int N = rows(K);
  matrix[N,N+1] R;
  matrix[N,N] Q = eigenvectors_sym(K);
  vector[N] l = eigenvalues_sym(K);

  int k = n_zero_eval(l); //number of zero eigen values
  // Coerce to [1, N-1] to keep slices valid and guarantee a constant null vector
  if (k < 1) k = 1;
  if (k > N-1) k = N-1;

  l = eval_zero(l, k); //replace eigenvalues with zero

  if (k > 1) {
    Q = GS_complete(Q, k);      // Rebuild the first k vectors (nullspace + possibly numerically-null dirs)
  } else {
    // Overwrite the null vector explicitly with the constant direction (unit-norm)
    Q[,1] = rep_vector(1/sqrt(N), N);
    // The remaining eigenvectors from eigenvectors_sym(K) for a centered K
    // lie in the mean-zero subspace, so they are already orthogonal to 1.
  }

  R[1:N,1:N] = Q;
  R[1:N,N+1] = l;
  return R;
}

/* Kernel related functions */
// Gram matrix for different kernels
real kernel_SE(vector x, vector y, real rho){
  // Output k(x,y) = cov(f(x), f(y)) where k is s.e. kernel
  // Input: x, y, and the length scale (rho) of s.e. kernel

  return exp(-squared_distance(x,y)/(2*square(rho)));
}
real kernel_exponential(vector x, vector y, real rho){
  // Output k(x,y) = cov(f(x), f(y)) where k is exponential kernel
  // Input: x, y, and the length scale (rho) of exponential kernel
  return exp(-distance(x,y)/(rho));
}
real kernel_AR(vector x, vector y, real rho){
  return((rho*distance(x,y))/(1-square(rho)));
}

real kernel_matern32(vector x, vector y, real rho){
  // Output k(x,y) = cov(f(x), f(y)) where k is matern kernel with kappa param=3/2
  // Input: x, y, and the length scale (rho)
  real r = (sqrt(3)*distance(x,y))/rho;
  return((1+r)*exp(-r));
}
real kernel_matern52(vector x, vector y, real rho){
  // Output k(x,y) = cov(f(x), f(y)) where k is matern kernel with kappa param=5/2
  // Input: x, y, and the length scale (rho)
  real r = (sqrt(5)*distance(x,y))/rho;
  return((1+r+r/3)*exp(-r));
}


real kernel_fBM(vector x, vector y, real Hurst){
  // Output k(x,y) = cov(f(x), f(y)) where k is fBM kernel with degree Hurst 
  // Input: x, y, and Hurst coefficient 
  real xysqdist = squared_distance(x,y);
  real xnorm2 = dot_self(x);
  real ynorm2 = dot_self(y);
  return(0.5*(pow(xnorm2, Hurst) + pow(ynorm2,Hurst) - pow(xysqdist, Hurst)));
  //return((xydist));
}

matrix Gram_SE(matrix X, real rho){
  // Output: n by n Gram matrix with squared exponential kernel
  // Input: X - predictor
  //        n - n_rows of X
  //        rho - length scale of s.e. kernel
  int N = rows(X);
  matrix[N,N] K = diag_matrix(rep_vector(1,N));
  for (i in 1:(N-1)){
    for (j in (i+1):N){
      K[i,j] = kernel_SE(to_vector(X[i,]),to_vector(X[j,]),rho);
      K[j,i] = K[i,j];
    }
  }
  return K ;
}

matrix Gram_exponential(matrix X, real rho){
  // Output: n by n Gram matrix with exponential kernel
  // Input: X - predictor
  //        n - n_rows of X
  //        rho - length scale of exponential kernel
  int N = rows(X);
  matrix[N,N] K = diag_matrix(rep_vector(1,N));
  for (i in 1:(N-1)){
    for (j in (i+1):N){
      K[i,j] = kernel_exponential(to_vector(X[i,]),to_vector(X[j,]),rho);
      K[j,i] = K[i,j];
    }
  }
  return K ;
}
matrix Gram_AR(matrix X, int N, real rho){
  // Output: n by n Gram matrix with exponential kernel
  // Input: X - predictor
  //        n - n_rows of X
  //        rho - correlation parameters of the kernel
  matrix[N,N] K = diag_matrix(rep_vector((1/(1-square(rho))),N));
  for (i in 1:(N-1)){
    for (j in (i+1):N){
      K[i,j] = kernel_AR(to_vector(X[i,]),to_vector(X[j,]),rho);
      K[j,i] = K[i,j];
    }
  }
  return K ;
}

matrix Gram_matern32(matrix X, real rho){
  // Output: n by n Gram matrix with matern kernel with kappa param=3/2
  // Input: X - predictor
  //        rho - length scale of the matern kernel
  int N = rows(X);
  matrix[N,N] K = diag_matrix(rep_vector(1,N));
  for (i in 1:(N-1)){
    for (j in (i+1):N){
      K[i,j] = kernel_matern32(to_vector(X[i,]),to_vector(X[j,]),rho);
      K[j,i] = K[i,j];
    }
  }
  return K ;
}
matrix Gram_matern52(matrix X, real rho){
  // Output: n by n Gram matrix with matern kernel with kappa param=5/2
  // Input: X - predictor
  //        rho - length scale of the matern kernel
  int N = rows(X);
  matrix[N,N] K = diag_matrix(rep_vector(1,N));
  for (i in 1:(N-1)){
    for (j in (i+1):N){
      K[i,j] = kernel_matern52(to_vector(X[i,]),to_vector(X[j,]),rho);
      K[j,i] = K[i,j];
    }
  }
  return K ;
}


matrix Gram_fBM(matrix X, real Hurst){
  // Output: Gram matrix with fractional brownian motion
  // Input: X - predictor
  //        N - nrow of X
  //        Hurst - Hurst coefficeint of fBM kernel
  matrix[rows(X), rows(X)] K;
  int N = rows(X);
  {
    vector[N] d;
    matrix[N, N] B = rep_matrix(0, N, N);;
    matrix[N, N] A = diag_matrix(rep_vector(1,N)) - (1.0/N)*rep_matrix(1, N, N);
    matrix[N, N] Xcp = X * X' ;
    vector[N] dvec = diagonal(Xcp);
    for (i in 1:(N-1)){
      d[i] = pow(abs(dvec[i]), Hurst);
      for (j in (i+1):N){
        B[i,j] = pow(abs(dvec[i] + dvec[j] - 2 * Xcp[i,j]), Hurst);
        B[j,i] = B[i,j];
      }
    }
    d[N] = pow(abs(dvec[N]), Hurst);
    {
      matrix[N, N] E = rep_matrix(d, N);
      K = 0.5 * (E + E' - B);
    }
  }
  return K ;
}

matrix Gram_fBM_sq_cen(matrix X, real Hurst){
  // Output: Gram matrix with square centered fractional brownian motion
  // Input: X - predictor
  //        Hurst - Hurst coefficeint of fBM kernel
  int N = rows(X);
  matrix[N, N] K = Gram_fBM(X, Hurst);
  matrix[N, N] A = diag_matrix(rep_vector(1,N)) - (1.0/N)*rep_matrix(1, N, N);
  K = A * K * A;
  K =  K * K;
  // to ensure that it's symmetric
  K = 0.5 *(K + K');
  return K;
}

matrix Gram_centring(matrix K){
    // Output: centered Gram matrix
    // Input: K - Uncentered Gram matrix
    int N = rows(K);
    matrix[N, N] A = diag_matrix(rep_vector(1,N)) - (1.0/N)*rep_matrix(1, N, N);
    return A * K * A ;
}

matrix Gram_OAK(matrix K){
    int N = rows(K);
    vector[N] g = K*rep_vector(1,N);
    return (K-(1/sum(g))*(g*g')) ;
}

matrix Gram_square(matrix K){
    // Output: squared Gram matrix
    // Input: K -  Gram matrix
    //        N - nrow of K
    int N = rows(K);
    matrix[N,N] Ksq = K * K ;
    Ksq = 0.5 *(Ksq + Ksq');
    return Ksq;
}

vector kvec_SE(matrix X, vector x_tes, real rho){
  // Output: a vector of k(x*, x_1),...,k(x*, x_N) for a given test point x*
  //          with squared exponential kernel
  // Input: X - (traing) predictor
  //        x_tes - test point
  //        N - n_rows of X
  //        rho - length scale of s.e. kernel
  vector[rows(X)] kvec ;
  int N = rows(X);
  for (i in 1:N){
    kvec[i] = kernel_SE(x_tes, to_vector(X[i,]), rho);
  }
  return kvec ;
}
vector kvec_matern52(matrix X, vector x_tes, real rho){
  // Output: a vector of k(x*, x_1),...,k(x*, x_N) for a given test point x*
  //          with squared exponential kernel
  // Input: X - (traing) predictor
  //        x_tes - test point
  //        N - n_rows of X
  //        rho - length scale of s.e. kernel
  vector[rows(X)] kvec ;
  int N = rows(X);
  for (i in 1:N){
    kvec[i] = kernel_matern52(x_tes, to_vector(X[i,]), rho);
  }
  return kvec ;
}

vector kvec_exponential(matrix X, vector x_tes, real rho){
  // Output: a vector of k(x*, x_1),...,k(x*, x_N) for a given test point x*
  //          with exponential kernel
  // Input: X - (traing) predictor
  //        x_tes - test point
  //        N - n_rows of X
  //        rho - length scale of the kernel
  vector[rows(X)] kvec ;
  int N = rows(X);
  for (i in 1:N){
    kvec[i] = kernel_exponential(x_tes, to_vector(X[i,]), rho);
  }
  return kvec ;
}

vector kvec_AR(matrix X, vector x_tes, real rho){
  // Output: a vector of k(x*, x_1),...,k(x*, x_N) for a given test point x*
  //          with AR
  // Input: X - (traing) predictor
  //        x_tes - test point
  //        N - n_rows of X
  //        rho - correlation parameters of the kernel
  vector[rows(X)] kvec ;
  int N = rows(X);
  for (i in 1:N){
    kvec[i] = kernel_AR(x_tes, to_vector(X[i,]), rho);
  }
  return kvec ;
}

vector kvec_fBM(matrix X, vector x_new, real Hurst){
  // Output: a vector of k(x*, x_1),...,k(x*, x_N) for a given test point x*
  //          with fBM kernel
  // Input: X - (traing) predictor
  //        x_tes - test point
  //        N - n_rows of X
  //        Hurst - hurst coefficeint of fBM kernel
  vector[rows(X)] kvec ;
  int N = rows(X);
  real t1 = pow(dot_self(x_new), Hurst) ;
  vector[N] t2 ;
  vector[N] t12 ;
  for (i in 1:N){
    t12[i] = pow(squared_distance(x_new,X[i,]), Hurst);
    t2[i] = pow(dot_self(X[i,]), Hurst);
  }
  kvec = 0.5*(rep_vector(t1,N) + t2 - t12) ;
  return kvec ;
}

vector kvec_cen(vector kvec, vector Krowsum){
  int N = num_elements(kvec) ;
  return kvec - rep_vector(sum(kvec)/N, N) -(1.0/N)*Krowsum + rep_vector(sum(Krowsum)/square(N), N);
}

vector kvec_OAK(vector kvec, vector Krowsum){
  return kvec - (sum(kvec)/sum(Krowsum))*Krowsum;
}

vector kvec_sq(vector kvec, matrix K){
  return K * kvec ;
}

real kstar_cen(real kstar, vector kvec1, vector kvec2, vector Krowsum){
  int N = num_elements(kvec1);
  return kstar - sum(kvec1)/N  - sum(kvec2)/N + sum(Krowsum)/square(N) ;
}

real kstar_OAK(real kstar, vector kvec1, vector kvec2, vector Krowsum){
  return kstar - sum(kvec1)*sum(kvec2)/sum(Krowsum);
}

matrix kronecker_prod(matrix A, matrix B) {
  matrix[rows(A) * rows(B), cols(A) * cols(B)] C;
  int m;
  int n;
  int p;
  int q;
  m = rows(A);
  n = cols(A);
  p = rows(B);
  q = cols(B);
  for (i in 1:m) {
    for (j in 1:n) {
      int row_start;
      int row_end;
      int col_start;
      int col_end;
      row_start = (i - 1) * p + 1;
      row_end = (i - 1) * p + p;
      col_start = (j - 1) * q + 1;
      col_end = (j - 1) * q + q;
      C[row_start:row_end, col_start:col_end] = A[i, j] * B;
    }
  }
  return C;
}
matrix mat_mat_prod(matrix A, matrix B, matrix C){
  // compute M = (A otimes B)%*%C
  matrix[rows(A)*rows(B),cols(C)] M;
  //int nrowA = rows(A);
  int ncolA = cols(A);
  //int nrowB = rows(B);
  int ncolB = cols(B);
  int nrowC = rows(C); // should equal ncolA * ncolB
  int ncolC = cols(C);
  for (i in 1:ncolC){
    vector[nrowC] c = C[,i];
    M[,i] = to_vector(B * (to_matrix(c, ncolB, ncolA) * A'));
  }
  return M;
}
}
