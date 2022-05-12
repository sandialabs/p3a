#pragma once

#include <cfloat>

#include "p3a_macros.hpp"

namespace p3a {

namespace constants {

template <class T>
struct maximum {
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline static constexpr
  T value() { return T::maximum(); }
};

template <>
struct maximum<double> {
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline static constexpr
  double value() { return DBL_MAX; }
};

template <class T>
struct minimum {
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline static constexpr
  T value() { return T::minimum(); }
};

template <>
struct minimum<double> {
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline static constexpr
  double value() { return -DBL_MAX; }
};

template <class T>
struct epsilon {
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline static constexpr
  T value() { return T::epsilon(); }
};

template <>
struct epsilon<float> {
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline static constexpr
  float value() { return FLT_EPSILON; }
};

template <>
struct epsilon<double> {
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline static constexpr
  double value() { return DBL_EPSILON; }
};

template <class T>
struct zero {
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline static constexpr
  T value() { return T::zero(); }
};

template <>
struct zero<int> {
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline static constexpr
  int value() { return 0; }
};

template <>
struct zero<float> {
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline static constexpr
  float value() { return 0.0f; }
};

template <>
struct zero<double> {
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline static constexpr
  double value() { return 0.0; }
};

}

template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
T pi_value() { return T(3.14159265358979323846264338327950288419716939937510l); }
template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
T maximum_value() { return constants::maximum<T>::value(); }
template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
T minimum_value() { return constants::minimum<T>::value(); }
template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
T epsilon_value() { return constants::epsilon<T>::value(); }
template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
T zero_value() { return constants::zero<T>::value(); }
template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
T one_value() { return T(1); }
template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
T speed_of_light_value() { return T(299792458); }
template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
T electron_volt_value() { return T(1.602176634e-19); }
template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
T boltzmann_value() { return T(1.380649e-23); }
template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
T inch_value() { return T(25.4e-3); }
template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
T square_root_of_two_value() { return T(1.41421356237309504880); }

}
