#pragma once

namespace p3a {

template <class T>
struct is_scalar_helper {
  inline static constexpr bool value = false;
};

template <>
struct is_scalar_helper<int> {
  inline static constexpr bool value = true;
};

template <>
struct is_scalar_helper<float> {
  inline static constexpr bool value = true;
};

template <>
struct is_scalar_helper<double> {
  inline static constexpr bool value = true;
};

template <class T>
inline constexpr bool is_scalar = is_scalar_helper<T>::value;

}
