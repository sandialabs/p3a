#pragma once

#include "p3a_vector3.hpp"
#include "p3a_symmetric3x3.hpp"

namespace p3a {

template <class T>
class skew3x3 {
  T m_xy;
  T m_xz;
  T m_yz;
 public:
  P3A_ALWAYS_INLINE constexpr
  skew3x3() = default;
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  skew3x3(
      T const& a,
      T const& b,
      T const& c)
    :m_xy(a)
    ,m_xz(b)
    ,m_yz(c)
  {
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T xx() const { return zero_value<T>; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& xy() const { return m_xy; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& xy() { return m_xy; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& xz() const { return m_xz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& xz() { return m_xz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T yx() const { return -m_xy; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T yy() const { return zero_value<T>; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& yz() const { return m_yz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& yz() { return m_yz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T zx() const { return -m_xz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T zy() const { return -m_yz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T zz() const { return zero_value<T>; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
  skew3x3 zero()
  {
    return skew3x3(T(0), T(0), T(0));
  }
};

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
skew3x3<T> cross_product_matrix(vector3<T> const& a)
{
  return skew3x3<T>(-a.z(), a.y(), -a.x());
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto square(skew3x3<T> const& a)
{
  using C = decltype(a.xy() * a.xy());
  auto const xy_xy = square(a.xy());
  auto const xz_xz = square(a.xz());
  auto const yz_yz = square(a.yz());
  return symmetric3x3<C>(
      -xy_xy - xz_xz,
      -(a.xz() * a.yz()),
      a.xy() * a.yz(),
      -xy_xy - yz_yz,
      -(a.xy() * a.xz()),
      -xz_xz - yz_yz);
}

template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, skew3x3<decltype(A() * B())>>::type
operator*(
    skew3x3<A> const& a,
    B const& b)
{
  return skew3x3<decltype(a.xy() * b)>(
      a.xy() * b,
      a.xz() * b,
      a.yz() * b);
}

template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<A>, skew3x3<decltype(A() * B())>>::type
operator*(
    A const& a,
    skew3x3<B> const& b)
{
  return b * a;
}

}
