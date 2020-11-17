#pragma once

#include <cmath>

#include "p3a_macros.hpp"

namespace p3a {

template <class T>
[[nodiscard]] CPL_ALWAYS_INLINE constexpr
auto square(T const& a)
{
  return a * a;
}

template <class T>
[[nodiscard]] CPL_ALWAYS_INLINE constexpr
auto cube(T const& a)
{
  return a * a * a;
}

template <class T>
[[nodiscard]] CPL_ALWAYS_INLINE constexpr
T average(T const& a, T const& b)
{
  return (a + b) / 2;
}

[[nodiscard]] CPL_ALWAYS_INLINE inline
double absolute_value(double a)
{
  return std::abs(a);
}

[[nodiscard]] CPL_ALWAYS_INLINE inline
double ceiling(double a)
{
  return std::ceil(a);
}

[[nodiscard]] CPL_ALWAYS_INLINE inline
double square_root(double a)
{
  return std::sqrt(a);
}

[[nodiscard]] CPL_ALWAYS_INLINE inline
double natural_exponential(double a)
{
  return std::exp(a);
}

[[nodiscard]] CPL_ALWAYS_INLINE inline
double exponentiate(double a, double b)
{
  return std::pow(a, b);
}

template <class T>
[[nodiscard]] CPL_ALWAYS_INLINE constexpr T const&
condition(bool a, T const& b, T const& c)
{
  return a ? b : c;
}

template <class T>
[[nodiscard]] CPL_ALWAYS_INLINE constexpr T const&
minimum(T const& a, T const& b)
{
  return condition(b < a, b, a);
}

template <class T>
[[nodiscard]] CPL_ALWAYS_INLINE constexpr T const&
maximum(T const& a, T const& b)
{
  return condition(a < b, b, a);
}

}
