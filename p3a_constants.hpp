#pragma once

#include <cfloat>

#include "p3a_macros.hpp"

namespace p3a {

namespace constants {

template <class T>
struct maximum;
template <>
struct maximum<double> {
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline static constexpr
  double value() { return DBL_MAX; }
};

template <class T>
struct minimum;
template <>
struct minimum<double> {
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline static constexpr
  double value() { return -DBL_MAX; }
};

}

template <class T>
struct zero_value_helper {
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline static constexpr
  T value() { return T::zero(); }
};

template <>
struct zero_value_helper<int> {
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline static constexpr
  int value() { return 0; }
};

template <>
struct zero_value_helper<float> {
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline static constexpr
  float value() { return 0.0f; }
};

template <>
struct zero_value_helper<double> {
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline static constexpr
  double value() { return 0.0; }
};

template <class T>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
double pi_value() { return T(3.14159265358979323846264338327950288419716939937510l); }
template <class T>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
T maximum_value() { return constants::maximum<T>::value(); }
template <class T>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
T minimum_value() { return constants::minimum<T>::value(); }
template <class T>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
T zero_value() { return zero_value_helper<T>::value(); }
template <class T>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
T one_value() { return T(1); }

}
