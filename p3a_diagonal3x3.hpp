#pragma once

#include "p3a_macros.hpp"
#include "p3a_scaled_identity3x3.hpp"

namespace p3a {

template <class T>
class diagonal3x3 {
  T m_xx;
  T m_yy;
  T m_zz;
 public:
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  diagonal3x3(
      T const& a,
      T const& b,
      T const& c)
    :m_xx(a)
    ,m_yy(b)
    ,m_zz(c)
  {}
  P3A_ALWAYS_INLINE inline
  diagonal3x3() = default;
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& xx() const { return m_xx; }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  T& xx() { return m_xx; }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& yy() const { return m_yy; }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  T& yy() { return m_yy; }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& zz() const { return m_zz; }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  T& zz() { return m_zz; }
};

template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
auto trace(diagonal3x3<T> const& d)
{
  return d.xx() + d.yy() + d.zz();
}

template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
auto determinant(diagonal3x3<T> const& d)
{
  return d.xx() * d.yy() * d.zz();
}

template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
diagonal3x3<T> operator-(
    diagonal3x3<T> const& a,
    scaled_identity3x3<T> const& b)
{
  return diagonal3x3<T>(
      a.xx() - b.scale(),
      a.yy() - b.scale(),
      a.zz() - b.scale());
}

template <class T, class U>
P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline
diagonal3x3<T>& operator-=(diagonal3x3<T>& a, U const& b)
{
  a = a - b;
  return a;
}

}
