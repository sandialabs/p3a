#pragma once

#include "p3a_vector2.hpp"

namespace p3a {

template <class T>
class matrix2x2 {
  T m_xx;
  T m_xy;
  T m_yx;
  T m_yy;
 public:
  P3A_ALWAYS_INLINE constexpr matrix2x2() = default;
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  matrix2x2(
      T const& a, T const& b,
      T const& c, T const& d)
    :m_xx(a)
    ,m_xy(b)
    ,m_yx(c)
    ,m_yy(d)
  {
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& xx() const { return m_xx; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& xy() const { return m_xy; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& yx() const { return m_yx; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& yy() const { return m_yy; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& xx() { return m_xx; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& xy() { return m_xy; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& yx() { return m_yx; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& yy() { return m_yy; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
  matrix2x2<T> zero()
  {
    return matrix2x2<T>(
        T(0), T(0),
        T(0), T(0));
  }
};

template <class T>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
void operator+=(matrix2x2<T>& a, matrix2x2<T> const& b)
{
  a.xx() += b.xx();
  a.xy() += b.xy();
  a.yx() += b.yx();
  a.yy() += b.yy();
}

template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, matrix2x2<decltype(A() / B())>>::type
operator/(matrix2x2<A> const& a, B const& b)
{
  using result_type = decltype(a.xx() / b);
  return matrix2x2<result_type>(
      a.xx() / b, a.xy() / b,
      a.yx() / b, a.yy() / b);
}

template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(matrix2x2<A> const& a, vector2<B> const& b)
{
  using result_type = decltype(a.xx() * b.x());
  return vector2<result_type>(
      a.xx() * b.x() + a.xy() * b.y(),
      a.yx() * b.x() + a.yy() * b.y());
}

template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto outer_product(vector2<A> const& a, vector2<B> const& b)
{
  using result_type = decltype(a.x() * b.x());
  return matrix2x2<result_type>(
      a.x() * b.x(), a.x() * b.y(),
      a.y() * b.x(), a.y() * b.y());
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto determinant(matrix2x2<T> const& m)
{
  T const& a = m.xx();
  T const& b = m.xy();
  T const& c = m.yx();
  T const& d = m.yy();
  return (a * d) - (b * c);
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto inverse(matrix2x2<T> const& m)
{
  T const& a = m.xx();
  T const& b = m.xy();
  T const& c = m.yx();
  T const& d = m.yy();
  return matrix2x2<T>(
      d, -b,
     -c,  a) / determinant(m);
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
matrix2x2<T> transpose(matrix2x2<T> const& m)
{
  return matrix2x2<T>(
      m.xx(), m.yx(),
      m.xy(), m.yy());
}

template <class A, class B>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
auto operator*(
    matrix2x2<A> const& a,
    matrix2x2<B> const& b)
{
  using C = decltype(a.xx() * b.xx());
  return matrix2x2<C>(
      a.xx() * b.xx() + a.xy() * b.yx(),
      a.xx() * b.xy() + a.xy() * b.yy(),
      a.yx() * b.xx() + a.yy() * b.yx(),
      a.yx() * b.xy() + a.yy() * b.yy());
}

}
