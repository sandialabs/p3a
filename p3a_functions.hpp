#pragma once

#include <cmath>

#include "p3a_macros.hpp"

namespace p3a {

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto square(T const& a)
{
  return a * a;
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto cube(T const& a)
{
  return a * a * a;
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
T average(T const& a, T const& b)
{
  return (a + b) / 2;
}

[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline
double absolute_value(double a)
{
  return std::abs(a);
}

[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline
double ceiling(double a)
{
  return std::ceil(a);
}

[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline
double square_root(double a)
{
  return std::sqrt(a);
}

[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline
double natural_exponential(double a)
{
  return std::exp(a);
}

[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline
double natural_logarithm(double a)
{
  return std::log(a);
}

[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline
double exponentiate(double a, double b)
{
  return std::pow(a, b);
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr T const&
condition(bool a, T const& b, T const& c)
{
  return a ? b : c;
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr auto
minimum(T const& a, T const& b)
{
  return condition(b < a, b, a);
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr auto
maximum(T const& a, T const& b)
{
  return condition(a < b, b, a);
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
T ceildiv(T a, T b) {
  return (a / b) + ((a % b) ? 1 : 0);
}

template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
A linear_interpolation(A a, A b, B t) {
  return a + t * (b - a);
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE
T load(T const* ptr, int offset)
{
  return ptr[offset];
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE
void store(T const& value, T* ptr, int offset)
{
  ptr[offset] = value;
}

}
