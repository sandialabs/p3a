#pragma once

#include "p3a_identity3x3.hpp"
#include "p3a_symmetric3x3.hpp"
#include "p3a_diagonal3x3.hpp"

namespace p3a {

template <class T>
class matrix3x3 {
  T m_xx;
  T m_xy;
  T m_xz;
  T m_yx;
  T m_yy;
  T m_yz;
  T m_zx;
  T m_zy;
  T m_zz;
 public:
  P3A_ALWAYS_INLINE constexpr matrix3x3() = default;
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  matrix3x3(
      T const& a, T const& b, T const& c,
      T const& d, T const& e, T const& f,
      T const& g, T const& h, T const& i)
    :m_xx(a)
    ,m_xy(b)
    ,m_xz(c)
    ,m_yx(d)
    ,m_yy(e)
    ,m_yz(f)
    ,m_zx(g)
    ,m_zy(h)
    ,m_zz(i)
  {
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& xx() const { return m_xx; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& xy() const { return m_xy; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& xz() const { return m_xz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& yx() const { return m_yx; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& yy() const { return m_yy; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& yz() const { return m_yz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& zx() const { return m_zx; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& zy() const { return m_zy; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& zz() const { return m_zz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& xx() { return m_xx; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& xy() { return m_xy; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& xz() { return m_xz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& yx() { return m_yx; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& yy() { return m_yy; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& yz() { return m_yz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& zx() { return m_zx; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& zy() { return m_zy; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& zz() { return m_zz; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
  matrix3x3<T> zero()
  {
    return matrix3x3<T>(
        T(0), T(0), T(0),
        T(0), T(0), T(0),
        T(0), T(0), T(0));
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
  matrix3x3<T> identity()
  {
    return matrix3x3<T>(
        T(1), T(0), T(0),
        T(0), T(1), T(0),
        T(0), T(0), T(1));
  }
};

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
symmetric3x3<T> symmetric(matrix3x3<T> const& a)
{
  return symmetric3x3<T>(
      a.xx(), 
      average(a.xy(), a.yx()),
      average(a.xz(), a.zx()),
      a.yy(),
      average(a.yz(), a.zy()),
      a.zz());
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
matrix3x3<T> operator+(matrix3x3<T> const& a, matrix3x3<T> const& b)
{
  return matrix3x3<T>(
      a.xx() + b.xx(),
      a.xy() + b.xy(),
      a.xz() + b.xz(),
      a.yx() + b.yx(),
      a.yy() + b.yy(),
      a.yz() + b.yz(),
      a.zx() + b.zx(),
      a.zy() + b.zy(),
      a.zz() + b.zz());
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
matrix3x3<T> operator+(identity3x3_type const&, matrix3x3<T> const& b)
{
  return matrix3x3<T>(
      T(1) + b.xx(),
      b.xy(),
      b.xz(),
      b.yx(),
      T(1) + b.yy(),
      b.yz(),
      b.zx(),
      b.zy(),
      T(1) + b.zz());
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
matrix3x3<T> operator+(matrix3x3<T> const& a, identity3x3_type const& b)
{
  return b + a;
}

template <class T>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
void operator+=(matrix3x3<T>& a, matrix3x3<T> const& b)
{
  a = a + b;
}

template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, matrix3x3<decltype(A() / B())>>::type
operator/(matrix3x3<A> const& a, B const& b)
{
  using result_type = decltype(a.xx() / b);
  return matrix3x3<result_type>(
      a.xx() / b, a.xy() / b, a.xz() / b,
      a.yx() / b, a.yy() / b, a.yz() / b,
      a.zx() / b, a.zy() / b, a.zz() / b);
}

template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(matrix3x3<A> const& a, vector3<B> const& b)
{
  using result_type = decltype(a.xx() * b.x());
  return vector3<result_type>(
      a.xx() * b.x() + a.xy() * b.y() + a.xz() * b.z(),
      a.yx() * b.x() + a.yy() * b.y() + a.yz() * b.z(),
      a.zx() * b.x() + a.zy() * b.y() + a.zz() * b.z());
}

template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto outer_product(vector3<A> const& a, vector3<B> const& b)
{
  using result_type = decltype(a.x() * b.x());
  return matrix3x3<result_type>(
      a.x() * b.x(), a.x() * b.y(), a.x() * b.z(),
      a.y() * b.x(), a.y() * b.y(), a.y() * b.z(),
      a.z() * b.x(), a.z() * b.y(), a.z() * b.z());
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto determinant(matrix3x3<T> const& m)
{
  T const& a = m.xx();
  T const& b = m.xy();
  T const& c = m.xz();
  T const& d = m.yx();
  T const& e = m.yy();
  T const& f = m.yz();
  T const& g = m.zx();
  T const& h = m.zy();
  T const& i = m.zz();
  return
    (a * e * i)
  + (b * f * g)
  + (c * d * h)
  - (c * e * g)
  - (b * d * i)
  - (a * f * h);
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto inverse(matrix3x3<T> const& m)
{
  T const& a = m.xx();
  T const& b = m.xy();
  T const& c = m.xz();
  T const& d = m.yx();
  T const& e = m.yy();
  T const& f = m.yz();
  T const& g = m.zx();
  T const& h = m.zy();
  T const& i = m.zz();
  auto const A =  (e * i - f * h);
  auto const B = -(d * i - f * g);
  auto const C =  (d * h - e * g);
  auto const D = -(b * i - c * h);
  auto const E =  (a * i - c * g);
  auto const F = -(a * h - b * g);
  auto const G =  (b * f - c * e);
  auto const H = -(a * f - c * d);
  auto const I =  (a * e - b * d);
  using result_type = std::remove_const_t<decltype(A)>;
  return matrix3x3<result_type>(
      A, D, G,
      B, E, H,
      C, F, I) / determinant(m);
}

template <class T>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
matrix3x3<T> load_matrix3x3(T const* ptr, int stride, int offset)
{
  return matrix3x3<T>(
      load(ptr, 0 * stride + offset),
      load(ptr, 1 * stride + offset),
      load(ptr, 2 * stride + offset),
      load(ptr, 3 * stride + offset),
      load(ptr, 4 * stride + offset),
      load(ptr, 5 * stride + offset),
      load(ptr, 6 * stride + offset),
      load(ptr, 7 * stride + offset),
      load(ptr, 8 * stride + offset));
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
void store(
    matrix3x3<T> const& value,
    T* ptr, int stride, int offset)
{
  store(value.xx(), ptr, 0 * stride + offset);
  store(value.xy(), ptr, 1 * stride + offset);
  store(value.xz(), ptr, 2 * stride + offset);
  store(value.yx(), ptr, 3 * stride + offset);
  store(value.yy(), ptr, 4 * stride + offset);
  store(value.yz(), ptr, 5 * stride + offset);
  store(value.zx(), ptr, 6 * stride + offset);
  store(value.zy(), ptr, 7 * stride + offset);
  store(value.zz(), ptr, 8 * stride + offset);
}

template <class A, class B>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
auto multiply_at_b_a(
    matrix3x3<A> const& a,
    diagonal3x3<B> const& b)
{
  using C = decltype(a.xx() * b.xx() * a.xx());
  return symmetric3x3<C>(
      a.xx() * b.xx() * a.xx() + a.yx() * b.yy() * a.yx() + a.zx() * b.zz() * a.zx(),
      a.xx() * b.xx() * a.xy() + a.yx() * b.yy() * a.yy() + a.zx() * b.zz() * a.zy(),
      a.xx() * b.xx() * a.xz() + a.yx() * b.yy() * a.yz() + a.zx() * b.zz() * a.zz(),
      a.xy() * b.xx() * a.xy() + a.yy() * b.yy() * a.yy() + a.zy() * b.zz() * a.zy(),
      a.xy() * b.xx() * a.xz() + a.yy() * b.yy() * a.yz() + a.zy() * b.zz() * a.zz(),
      a.xz() * b.xx() * a.xz() + a.yz() * b.yy() * a.yz() + a.zz() * b.zz() * a.zz());
}

template <class A, class B>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
auto multiply_a_b_at(
    matrix3x3<A> const& a,
    diagonal3x3<B> const& b)
{
  using C = decltype(a.xx() * b.xx() * a.xx());
  return symmetric3x3<C>(
      a.xx() * b.xx() * a.xx() + a.xy() * b.yy() * a.xy() + a.xz() * b.zz() * a.xz(),
      a.xx() * b.xx() * a.yx() + a.xy() * b.yy() * a.yy() + a.xz() * b.zz() * a.yz(),
      a.xx() * b.xx() * a.zx() + a.xy() * b.yy() * a.zy() + a.xz() * b.zz() * a.zz(),
      a.yx() * b.xx() * a.yx() + a.yy() * b.yy() * a.yy() + a.yz() * b.zz() * a.yz(),
      a.yx() * b.xx() * a.zx() + a.yy() * b.yy() * a.zy() + a.yz() * b.zz() * a.zz(),
      a.zx() * b.xx() * a.zx() + a.zy() * b.yy() * a.zy() + a.zz() * b.zz() * a.zz());
}

template <class A, class B>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
auto operator*(
    matrix3x3<A> const& a,
    matrix3x3<B> const& b)
{
  using C = decltype(a.xx() * b.xx());
  return matrix3x3<C>(
      a.xx() * b.xx() + a.xy() * b.yx() + a.xz() * b.zx(),
      a.xx() * b.xy() + a.xy() * b.yy() + a.xz() * b.zy(),
      a.xx() * b.xz() + a.xy() * b.yz() + a.xz() * b.zz(),
      a.yx() * b.xx() + a.yy() * b.yx() + a.yz() * b.zx(),
      a.yx() * b.xy() + a.yy() * b.yy() + a.yz() * b.zy(),
      a.yx() * b.xz() + a.yy() * b.yz() + a.yz() * b.zz(),
      a.zx() * b.xx() + a.zy() * b.yx() + a.zz() * b.zx(),
      a.zx() * b.xy() + a.zy() * b.yy() + a.zz() * b.zy(),
      a.zx() * b.xz() + a.zy() * b.yz() + a.zz() * b.zz());
}

}
