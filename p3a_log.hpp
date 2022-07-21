#pragma once

// The logarithmic map computed by Padé approximants.

#include "p3a_matrix3x3.hpp"
#include "p3a_tensor_detail.hpp"

namespace p3a {

// Matrix square root by product form of Denman-Beavers iteration.
template <typename T>
[[nodiscard]] P3A_HOST_DEVICE inline auto
sqrt_dbp(matrix3x3<T> const& A, int& k)
{
  auto const eps      = epsilon_value<T>();
  auto const tol      = 0.5 * p3a::sqrt(3.0) * eps;  // 3 is dim
  auto const I        = matrix3x3<T>::identity();
  auto const max_iter = 32;
  auto       X        = A;
  auto       M        = A;
  auto       scale    = true;
  k                   = 0;
  while (k++ < max_iter) {
    if (scale == true) {
      auto const d  = p3a::abs(determinant(M));
      auto const d2 = p3a::sqrt(d);
      auto const d6 = p3a::cbrt(d2);
      auto const g  = 1.0 / d6;
      X *= g;
      M *= g * g;
    }
    auto const Y = X;
    auto const N = inverse(M);
    X *= 0.5 * (I + N);
    M                = 0.5 * (I + 0.5 * (M + N));
    auto const error = norm(M - I);
    auto const diff  = norm(X - Y) / norm(X);
    scale            = diff >= 0.01;
    if (error <= tol) break;
  }
  return X;
}

// Matrix square root
template <typename T>
[[nodiscard]] P3A_HOST_DEVICE inline auto
sqrt(matrix3x3<T> const& A)
{
  int i = 0;
  return sqrt_dbp(A, i);
}

// Logarithmic map by Padé approximant and partial fractions
template <typename T>
[[nodiscard]] P3A_HOST_DEVICE inline auto
log_pade_pf(matrix3x3<T> const& A, int const n)
{
  auto const I = matrix3x3<T>::identity();
  auto       X = 0.0 * A;
  for (auto i = 0; i < n; ++i) {
    auto const x = 0.5 * (1.0 + gauss_legendre_abscissae<T>(n, i));
    auto const w = 0.5 * gauss_legendre_weights<T>(n, i);
    auto const B = I + x * A;
    X += w * A * inverse_full_pivot(B);
  }
  return X;
}

// Logarithmic map by inverse scaling and squaring and Padé approximants
template <typename T>
[[nodiscard]] P3A_HOST_DEVICE inline auto
log_iss(matrix3x3<T> const& A)
{
  auto const I   = matrix3x3<T>::identity();
  auto const c15 = pade_coefficients<T>(15);
  auto       X   = A;
  auto       i   = 5;
  auto       j   = 0;
  auto       k   = 0;
  auto       m   = 0;
  while (true) {
    auto const diff = norm_1(X - I);
    if (diff <= c15) {
      auto p = 2;
      while (pade_coefficients<T>(p) <= diff && p < 16) {
        ++p;
      }
      auto q = 2;
      while (pade_coefficients<T>(q) <= diff / 2.0 && q < 16) {
        ++q;
      }
      if ((2 * (p - q) / 3) < i || ++j == 2) {
        m = p + 1;
        break;
      }
    }
    X = sqrt_dbp(X, i);
    ++k;
  }
  X = (1U << k) * log_pade_pf(X - I, m);
  return X;
}

// Logarithmic map
template <typename T>
[[nodiscard]] P3A_HOST_DEVICE inline auto
log(matrix3x3<T> const& A)
{
  return log_iss(A);
}

// Logarithm by Gregory series for verification. Convergence guaranteed for symmetric A
template <typename T>
[[nodiscard]] P3A_HOST_DEVICE inline auto
log_gregory(matrix3x3<T> const& A)
{
  auto const max_iter = 8192;
  auto const tol      = epsilon_value<T>();
  auto const I        = matrix3x3<T>::identity();
  auto const IpA      = I + A;
  auto const ImA      = I - A;
  auto       S        = ImA * inverse(IpA);
  auto       norm_s   = norm(S);
  auto const C        = S * S;
  auto       B        = S;
  auto       k        = 0;
  while (norm_s > tol && ++k <= max_iter) {
    S = (2.0 * k - 1.0) * S * C / (2.0 * k + 1.0);
    B += S;
    norm_s = norm(S);
  }
  B *= -2.0;
  return B;
}

} // namespace p3a
