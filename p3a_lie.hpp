#pragma once

// functions related to Lie algebras of tensors,
// in particular the tensor logarithm and exponential
// 
// Most of these functions are from the following publication:
// Mota, Alejandro, et al.
// "Lie-group interpolation and variational recovery for internal variables."
// Computational Mechanics 52.6 (2013): 1281-1299.

#include "p3a_eigen.hpp"

namespace p3a {

template <class T>
[[nodiscard]] P3A_HOST_DEVICE inline
diagonal3x3<T> log(diagonal3x3<T> const& m)
{
  return diagonal3x3<T>(
      p3a::log(m.xx()),
      p3a::log(m.yy()),
      p3a::log(m.zz()));
}

template <class T>
[[nodiscard]] P3A_HOST_DEVICE inline
diagonal3x3<T> exp(diagonal3x3<T> const& m)
{
  return diagonal3x3<T>(
      p3a::exp(m.xx()),
      p3a::exp(m.yy()),
      p3a::exp(m.zz()));
}

template <class T>
[[nodiscard]] P3A_HOST_DEVICE inline
symmetric3x3<T> log_positive_definite(symmetric3x3<T> const& S)
{
  diagonal3x3<T> D;
  matrix3x3<T> U;
  eigendecompose(S, D, U);
  return multiply_a_b_at(U, log(D));
}

template <class T>
[[nodiscard]] P3A_HOST_DEVICE inline
symmetric3x3<T> exp(symmetric3x3<T> const& H)
{
  diagonal3x3<T> L;
  matrix3x3<T> U;
  eigendecompose(H, L, U);
  return multiply_a_b_at(U, exp(L));
}

}
