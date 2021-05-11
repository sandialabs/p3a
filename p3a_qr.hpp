#pragma once

namespace p3a {

namespace qr {

enum class errc {
  success,
  rank_deficient
};

/* Trefethen, Lloyd N., and David Bau III.
   Numerical linear algebra. Vol. 50. SIAM, 1997.
   Algorithm 10.1. Householder QR Factorization

   for k=1 to n
     x = A_{k:m,k}
     v_k = sign(x_1)\|x\|_2 e_1 + x
     v_k = v_k / \|v_k\|_2          <- note this can divide by zero if x={0}
     A_{k:m,k:n} = A_{k:m,k:n} - 2 v_k (v_k^* A_{k:m,k:n}) */

template <class T, int max_m, int max_n>
P3A_HOST P3A_DEVICE errc householder_vector(
    int m,
    static_array<static_array<T, max_m>, max_n> const& a,
    int k,
    static_array<T, max_m>& v_k)
{
  T norm_x = 0.0;
  for (int i = k; i < m; ++i) norm_x += square(a[k][i]);
  norm_x = square_root(norm_x);
  /* technically, every matrix has a QR decomposition.
   * if norm_x is close to zero here, the matrix is rank-deficient
   * and we could just skip this reflection and carry forward
   * the rank information.
   * however, all current uses of this code require the matrix
   * to be full-rank, so we can save a bunch of bookkeeping up
   * the stack if we simply assert this here.
   */
  if (norm_x == 0.0) return errc::rank_deficient;
  for (int i = k; i < m; ++i) v_k[i] = a[k][i];
  v_k[k] += sign(a[k][k]) * norm_x;
  double norm_v_k = 0.0;
  for (int i = k; i < m; ++i) norm_v_k += square(v_k[i]);
  norm_v_k = square_root(norm_v_k);
  for (int i = k; i < m; ++i) v_k[i] /= norm_v_k;
  return errc::success;
}

template <class T, int max_m, int max_n>
P3A_HOST P3A_DEVICE void reflect_columns(
    int m,
    int n,
    static_array<static_array<T, max_m>, max_n>& a,
    static_array<T, max_m> const& v_k,
    int k)
{
  for (int j = k; j < n; ++j) {
    T dot = 0.0;
    for (int i = k; i < m; ++i) dot += a[j][i] * v_k[i];
    for (int i = k; i < m; ++i) a[j][i] -= 2.0 * dot * v_k[i];
  }
}

template <class T, int max_m, int max_n>
struct factorization {
  static_array<static_array<T, max_m>, max_n> v;  // the householder vectors
  static_array<static_array<T, max_n>, max_n> r;
};

template <class T, int max_m, int max_n>
P3A_HOST P3A_DEVICE errc factorize_qr_householder(
    int m,
    int n,
    static_array<static_array<T, max_m>, max_n>& a,
    factorization<T, max_m, max_n>& result)
{
  static_array<static_array<T, max_m>, max_n> v;
  for (int k = 0; k < n; ++k) {
    errc error = householder_vector(m, a, k, result.v[k]);
    if (error != errc::success) return error;
    reflect_columns(m, n, a, result.v[k], k);
  }
  reduced_r_from_full(n, a, result.r);
  return errc::success;
}

/* Trefethen, Lloyd N., and David Bau III.
   Numerical linear algebra. Vol. 50. SIAM, 1997.
   Algorithm 10.2. Implicit Calculation of a Product $Q^*b$

   for k=1 to n
     b_{k:m} = b_{k:m} - 2 v_k (v_k^* b_{k:m}) */

template <class T, int max_m, int max_n>
P3A_HOST P3A_DEVICE void implicit_q_trans_b(
    int m,
    int n,
    static_array<static_array<T, max_m>, max_n> const& v,
    static_array<T, max_m>& b) {
  for (int k = 0; k < n; ++k) {
    double dot = 0.0;
    for (int i = k; i < m; ++i) dot += v[k][i] * b[i];
    for (int i = k; i < m; ++i) b[i] -= 2.0 * dot * v[k][i];
  }
}

/* Trefethen, Lloyd N., and David Bau III.
   Numerical linear algebra. Vol. 50. SIAM, 1997.
   Algorithm 10.2. Implicit Calculation of a Product $Qx$

   for k=n downto 1
     x_{k:m} = x_{k:m} - 2 v_k (v_k^* b_{k:m}) */

template <int max_m, int max_n>
P3A_HOST P3A_DEVICE void implicit_q_x(
    int m, int n, static_array<double, max_m>& x, Few<static_array<double, max_m>, max_n> v) {
  for (int k2 = 0; k2 < n; ++k2) {
    int k = n - k2 - 1;
    double dot = 0;
    for (int i = k; i < m; ++i) dot += v[k][i] * x[i];
    for (int i = k; i < m; ++i) x[i] -= 2 * dot * v[k][i];
  }
}

template <class T, int max_m, int max_n>
P3A_HOST P3A_DEVICE void reduced_r_from_full(
    int n,
    static_array<static_array<T, max_m>, max_n> const& fr,
    static_array<static_array<T, max_n>, max_n>& rr)
{
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      rr[j][i] = fr[j][i];
    }
  }
}

template <class T, int max_m, int max_n>
P3A_HOST P3A_DEVICE void solve_upper_triangular(
    int m,
    static_array<static_array<T, max_m>, max_n> const& a,
    static_array<T, max_m> const& b,
    static_array<T, max_n>& x)
{
  for (int ii = 0; ii < m; ++ii) {
    int i = m - ii - 1;
    x[i] = b[i];
    for (int j = i + 1; j < m; ++j) x[i] -= a[j][i] * x[j];
    x[i] /= a[i][i];
  }
}

/* Trefethen, Lloyd N., and David Bau III.
   Numerical linear algebra. Vol. 50. SIAM, 1997.
   Algorithm 11.2 Least Squares via QR factorization

   1. Compute the reduced QR factorization A = \hat{Q}\hat{R}
   2. Compute the vector \hat{Q}^* b
   3. Solve the upper-triangular system \hat{R} x = \hat{Q}^* b for x  */

template <class T, int max_m, int max_n>
P3A_HOST P3A_DEVICE void solve(
    int m,
    int n,
    static_array<static_array<T, max_m>, max_n>& a,
    static_array<T, max_m>& b,
    static_array<T, max_n>& x)
{
  factorization<T, max_m, max_n> qr;
  errc error = factorize_qr_householder(m, n, a, qr);
  if (error != errc::success) return error;
  implicit_q_trans_b(m, n, qr.v, b);
  solve_upper_triangular(n, qr.r, q, x);
  return errc::success;
}

template <class T, int max_m, int max_n>
P3A_HOST P3A_DEVICE void solve(
    static_array<static_array<T, max_m>, max_n>& a,
    static_array<T, max_m>& b,
    static_array<T, max_n>& x)
{
  return solve(max_m, max_n, a, b);
}

}

}
