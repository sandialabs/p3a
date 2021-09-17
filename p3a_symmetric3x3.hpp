#pragma once

#include "p3a_macros.hpp"
#include "p3a_scaled_identity3x3.hpp"
#include "p3a_diagonal3x3.hpp"

namespace p3a {

template <class T>
class symmetric3x3 {
  T m_xx;
  T m_xy;
  T m_xz;
  T m_yy;
  T m_yz;
  T m_zz;
 public:
  P3A_ALWAYS_INLINE constexpr
  symmetric3x3() = default;
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  symmetric3x3(
      T const& a, T const& b, T const& c,
      T const& d, T const& e, T const& f)
    :m_xx(a)
    ,m_xy(b)
    ,m_xz(c)
    ,m_yy(d)
    ,m_yz(e)
    ,m_zz(f)
  {
  }
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  symmetric3x3(scaled_identity3x3<T> const& a)
    :symmetric3x3(a.scale(), T(0), T(0), a.scale(), T(0), a.scale())
  {}
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& xx() const { return m_xx; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& xy() const { return m_xy; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& xz() const { return m_xz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& yx() const { return m_xy; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& yy() const { return m_yy; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& yz() const { return m_yz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& zx() const { return m_xz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& zy() const { return m_yz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& zz() const { return m_zz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& xx() { return m_xx; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& xy() { return m_xy; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& xz() { return m_xz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& yx() { return m_xy; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& yy() { return m_yy; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& yz() { return m_yz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& zx() { return m_xz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& zy() { return m_yz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& zz() { return m_zz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
  symmetric3x3 zero()
  {
    return symmetric3x3<T>(
        T(0), T(0), T(0),
        T(0), T(0), T(0));
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
  symmetric3x3 identity()
  {
    return symmetric3x3<T>(
        T(1), T(0), T(0),
        T(1), T(0), T(1));
  }
};

template <class T>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
void operator+=(symmetric3x3<T>& a, symmetric3x3<T> const& b)
{
  a.xx() += b.xx();
  a.xy() += b.xy();
  a.xz() += b.xz();
  a.yy() += b.yy();
  a.yz() += b.yz();
  a.zz() += b.zz();
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
symmetric3x3<T> operator+(symmetric3x3<T> const& a, symmetric3x3<T> const& b)
{
  return symmetric3x3<T>(
    a.xx() + b.xx(),
    a.xy() + b.xy(),
    a.xz() + b.xz(),
    a.yy() + b.yy(),
    a.yz() + b.yz(),
    a.zz() + b.zz());
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
symmetric3x3<T> operator-(symmetric3x3<T> const& a, symmetric3x3<T> const& b)
{
  return symmetric3x3<T>(
    a.xx() - b.xx(),
    a.xy() - b.xy(),
    a.xz() - b.xz(),
    a.yy() - b.yy(),
    a.yz() - b.yz(),
    a.zz() - b.zz());
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
symmetric3x3<T> operator-(symmetric3x3<T> const& a)
{
  return symmetric3x3<T>(
    -a.xx(),
    -a.xy(),
    -a.xz(),
    -a.yy(),
    -a.yz(),
    -a.zz());
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
T trace(symmetric3x3<T> const& a)
{
  return a.xx() + a.yy() + a.zz();
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
symmetric3x3<T> operator+(
    symmetric3x3<T> const& a,
    scaled_identity3x3<T> const& b)
{
  return symmetric3x3<T>(
      a.xx() + b.scale(),
      a.xy(),
      a.xz(),
      a.yy() + b.scale(),
      a.yz(),
      a.zz() + b.scale());
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
symmetric3x3<T> operator-(
    scaled_identity3x3<T> const& a,
    symmetric3x3<T> const& b)
{
  return symmetric3x3<T>(
      a.scale() - b.xx(),
      -b.xy(),
      -b.xz(),
      a.scale() - b.yy(),
      -b.yz(),
      a.scale() - b.zz());
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
symmetric3x3<T> operator+(
    scaled_identity3x3<T> const& a,
    symmetric3x3<T> const& b)
{
  return b + a;
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
symmetric3x3<T> operator-(
    symmetric3x3<T> const& a,
    scaled_identity3x3<T> const& b)
{
  return symmetric3x3<T>(
      a.xx() - b.scale(),
      a.xy(),
      a.xz(),
      a.yy() - b.scale(),
      a.yz(),
      a.zz() - b.scale());
}

template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, symmetric3x3<decltype(A() * B())>>::type
operator*(
    symmetric3x3<A> const& a,
    B const& b)
{
  return symmetric3x3<decltype(a.xx() * b)>(
      a.xx() * b,
      a.xy() * b,
      a.xz() * b,
      a.yy() * b,
      a.yz() * b,
      a.zz() * b);
}

template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, symmetric3x3<decltype(A() / B())>>::type
operator/(
    symmetric3x3<A> const& a,
    B const& b)
{
  return symmetric3x3<decltype(a.xx() / b)>(
      a.xx() / b,
      a.xy() / b,
      a.xz() / b,
      a.yy() / b,
      a.yz() / b,
      a.zz() / b);
}

template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<A>, symmetric3x3<decltype(A() * B())>>::type
operator*(
    A const& a,
    symmetric3x3<B> const& b)
{
  return b * a;
}

template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(symmetric3x3<A> const& a, vector3<B> const& b)
{
  using C = decltype(a.xx() * b.x());
  return vector3<C>(
      a.xx() * b.x() + a.xy() * b.y() + a.xz() * b.z(),
      a.yx() * b.x() + a.yy() * b.y() + a.yz() * b.z(),
      a.zx() * b.x() + a.zy() * b.y() + a.zz() * b.z());
}

template <class A, class B>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto frobenius_inner_product(
    symmetric3x3<A> const& a, symmetric3x3<B> const& b)
{
  return 
       a.xx() * b.xx() +
       2 * a.xy() * b.xy() +
       2 * a.xz() * b.xz() +
       a.yy() * b.yy() +
       2 * a.yz() * b.yz() +
       a.zz() * b.zz();
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto outer_product(vector3<T> const& a)
{
  using result_type = decltype(a.x() * a.x());
  return symmetric3x3<result_type>(
      a.x() * a.x(), a.x() * a.y(), a.x() * a.z(),
      a.y() * a.y(), a.y() * a.z(),
      a.z() * a.z());
}

inline int constexpr symmetric3x3_component_count = 6;

template <class T>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
symmetric3x3<T> load_symmetric3x3(T const* ptr, int stride, int offset)
{
  return symmetric3x3<T>(
      load(ptr, 0 * stride + offset),
      load(ptr, 1 * stride + offset),
      load(ptr, 2 * stride + offset),
      load(ptr, 3 * stride + offset),
      load(ptr, 4 * stride + offset),
      load(ptr, 5 * stride + offset));
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
void store(
    symmetric3x3<T> const& value,
    T* ptr, int stride, int offset)
{
  store(value.xx(), ptr, 0 * stride + offset);
  store(value.xy(), ptr, 1 * stride + offset);
  store(value.xz(), ptr, 2 * stride + offset);
  store(value.yy(), ptr, 3 * stride + offset);
  store(value.yz(), ptr, 4 * stride + offset);
  store(value.zz(), ptr, 5 * stride + offset);
}

}
