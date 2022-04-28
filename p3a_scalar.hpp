#pragma once

namespace p3a {

namespace details {

template <class T>
struct is_scalar {
  inline static constexpr bool value = false;
};

template <>
struct is_scalar<int> {
  inline static constexpr bool value = true;
};

template <>
struct is_scalar<unsigned int> {
  inline static constexpr bool value = true;
};

template <>
struct is_scalar<float> {
  inline static constexpr bool value = true;
};

template <>
struct is_scalar<double> {
  inline static constexpr bool value = true;
};

}

template <class T>
inline constexpr bool is_scalar = details::is_scalar<T>::value;

}
