#pragma once

// The exponential map computed by Padé approximants.

#include "p3a_matrix3x3.hpp"
#include "p3a_tensor_detail.hpp"

namespace p3a {

// Padé approximant polynomial odd and even terms.
template <typename T>
P3A_HOST_DEVICE inline void
pade_polynomial_terms(matrix3x3<T> const& A, int const order, matrix3x3<T>& U, matrix3x3<T>& V)
{
  auto B        = matrix3x3<T>::identity();
  U             = polynomial_coefficient<T>(order, 1) * B;
  V             = polynomial_coefficient<T>(order, 0) * B;
  auto const A2 = A * A;
  for (int i = 3; i <= order; i += 2) {
    B            = B * A2;
    auto const O = polynomial_coefficient<T>(order, i) * B;
    auto const E = polynomial_coefficient<T>(order, i - 1) * B;
    U += O;
    V += E;
  }
  U = A * U;
}

// Compute a non-negative integer power of a tensor by binary manipulation.
template <typename T>
[[nodiscard]] P3A_HOST_DEVICE inline auto
binary_powering(matrix3x3<T> const& A, int const e)
{
  using bits               = uint64_t;
  bits const number_digits = 64;
  bits const exponent      = static_cast<bits>(e);
  if (exponent == 0) return matrix3x3<T>::identity();
  bits const rightmost_bit = 1;
  bits const leftmost_bit  = rightmost_bit << (number_digits - 1);
  bits       t             = 0;
  for (bits j = 0; j < number_digits; ++j) {
    if (((exponent << j) & leftmost_bit) != 0) {
      t = number_digits - j - 1;
      break;
    }
  }
  auto P = A;
  bits i = 0;
  bits m = exponent;
  while ((m & rightmost_bit) == 0) {
    P = P * P;
    ++i;
    m = m >> 1;
  }
  auto X = P;
  for (bits j = i + 1; j <= t; ++j) {
    P = P * P;
    if (((exponent >> j) & rightmost_bit) != 0) {
      X = X * P;
    }
  }
  return X;
}

// Exponential map by squaring and scaling and Padé approximants.
// See algorithm 10.20 in Functions of Matrices, N.J. Higham, SIAM, 2008.
template <typename T>
[[nodiscard]] P3A_HOST_DEVICE inline auto
exp(matrix3x3<T> const& A)
{
  auto       B             = matrix3x3<T>::identity();
  int const  orders[]      = {3, 5, 7, 9, 13};
  auto const number_orders = 5;
  auto const highest_order = orders[number_orders - 1];
  auto const norm          = norm_1(A);
  for (auto i = 0; i < number_orders; ++i) {
    auto const order = orders[i];
    auto const theta = scaling_squaring_theta<T>(order);
    if (order < highest_order && norm < theta) {
      auto U = B;
      auto V = B;
      pade_polynomial_terms(A, order, U, V);
      B = inverse(V - U) * (U + V);
      break;
    } else if (order == highest_order) {
      auto const theta_highest = scaling_squaring_theta<T>(order);
      auto const signed_power  = static_cast<int>(std::ceil(std::log2(norm / theta_highest)));
      auto const power_two     = signed_power > 0 ? static_cast<int>(signed_power) : 0;
      auto       scale         = 1.0;
      for (int j = 0; j < power_two; ++j) {
        scale /= 2.0;
      }
      auto const I        = matrix3x3<T>::identity();
      auto const A1       = scale * A;
      auto const A2       = A1 * A1;
      auto const A4       = A2 * A2;
      auto const A6       = A2 * A4;
      auto const b0       = polynomial_coefficient<T>(order, 0);
      auto const b1       = polynomial_coefficient<T>(order, 1);
      auto const b2       = polynomial_coefficient<T>(order, 2);
      auto const b3       = polynomial_coefficient<T>(order, 3);
      auto const b4       = polynomial_coefficient<T>(order, 4);
      auto const b5       = polynomial_coefficient<T>(order, 5);
      auto const b6       = polynomial_coefficient<T>(order, 6);
      auto const b7       = polynomial_coefficient<T>(order, 7);
      auto const b8       = polynomial_coefficient<T>(order, 8);
      auto const b9       = polynomial_coefficient<T>(order, 9);
      auto const b10      = polynomial_coefficient<T>(order, 10);
      auto const b11      = polynomial_coefficient<T>(order, 11);
      auto const b12      = polynomial_coefficient<T>(order, 12);
      auto const b13      = polynomial_coefficient<T>(order, 13);
      auto const U        = A1 * ((A6 * (b13 * A6 + b11 * A4 + b9 * A2) + b7 * A6 + b5 * A4 + b3 * A2 + b1 * I));
      auto const V        = A6 * (b12 * A6 + b10 * A4 + b8 * A2) + b6 * A6 + b4 * A4 + b2 * A2 + b0 * I;
      auto const R        = inverse(V - U) * (U + V);
      auto const exponent = (1 << power_two);
      B                   = binary_powering(R, exponent);
    }
  }
  return B;
}

// Exponential map by power series for verification, radius of convergence is infinity
template <typename T>
[[nodiscard]] P3A_HOST_DEVICE inline auto
exp_taylor(matrix3x3<T> const& A)
{
  auto const max_iter = 1024;
  auto const tol      = epsilon_value<T>();
  auto       term     = matrix3x3<T>::identity();
  // Relative error taken wrt to the first term, which is I and norm = 1
  auto relative_error = 1.0;
  auto B              = term;
  auto k              = 0;
  while (relative_error > tol && k < max_iter) {
    term           = static_cast<T>(1.0 / (k + 1.0)) * term * A;
    B              = B + term;
    relative_error = norm_1(term);
    ++k;
  }
  return B;
}

} // namespace p3a