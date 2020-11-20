#pragma once

#include "p3a_macros.hpp"
#include "p3a_identity3x3.hpp"

namespace p3a {

template <class T>
class scaled_identity3x3 {
  T m_scale;
 public:
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  scaled_identity3x3(T const& a)
    :m_scale(a)
  {}
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& scale() const { return m_scale; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T xx() const { return m_scale; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T xy() const { return zero_value<T>; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T xz() const { return zero_value<T>; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T yx() const { return zero_value<T>; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T yy() const { return m_scale; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T yz() const { return zero_value<T>; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T zx() const { return zero_value<T>; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T zy() const { return zero_value<T>; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T zz() const { return m_scale; }
};

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<T>, scaled_identity3x3<T>>::type
operator*(T const& a, identity3x3_type) {
  return scaled_identity3x3<T>(a);
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<T>, scaled_identity3x3<T>>::type
operator*(identity3x3_type, T const& a) {
  return scaled_identity3x3<T>(a);
}

template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<A>, scaled_identity3x3<decltype(A() * B())>>::type
operator*(
    A const& a,
    scaled_identity3x3<B> const& b)
{
  return scaled_identity3x3<decltype(a * b.scale())>(a * b.scale());
}

}
