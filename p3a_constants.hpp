#pragma once

#include <cfloat>

#include "p3a_macros.hpp"

namespace p3a {

namespace constants {

template <class T>
struct maximum;
template <>
struct maximum<double> {
  static constexpr double value = DBL_MAX;
};

template <class T>
struct minimum;
template <>
struct minimum<double> {
  static constexpr double value = -DBL_MAX;
};

}

template <class T>
struct zero_value_helper {
  inline static constexpr T value = T::zero();
};

template <>
struct zero_value_helper<int> {
  inline static constexpr int value = 0;
};

template <>
struct zero_value_helper<float> {
  inline static constexpr float value = 0.0f;
};

template <>
struct zero_value_helper<double> {
  inline static constexpr double value = 0.0;
};

template <class T> inline double constexpr pi_value = T(3.14159265358979323846264338327950288419716939937510l);
template <class T> inline T constexpr maximum_value = constants::maximum<T>::value;
template <class T> inline T constexpr minimum_value = constants::minimum<T>::value;
template <class T> inline T constexpr zero_value = zero_value_helper<T>::value;
template <class T> inline T constexpr one_value = T(1);

}
