#pragma once

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
  CPL_ALWAYS_INLINE constexpr matrix3x3() = default;
  CPL_ALWAYS_INLINE constexpr
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
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T const& xx() const { return m_xx; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T const& xy() const { return m_xy; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T const& xz() const { return m_xz; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T const& yx() const { return m_yx; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T const& yy() const { return m_yy; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T const& yz() const { return m_yz; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T const& zx() const { return m_zx; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T const& zy() const { return m_zy; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T const& zz() const { return m_zz; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T& xx() { return m_xx; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T& xy() { return m_xy; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T& xz() { return m_xz; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T& yx() { return m_yx; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T& yy() { return m_yy; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T& yz() { return m_yz; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T& zx() { return m_zx; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T& zy() { return m_zy; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T& zz() { return m_zz; }
  [[nodiscard]] CPL_ALWAYS_INLINE static constexpr
  matrix3x3<T> zero()
  {
    return matrix3x3<T>(
        T(0), T(0), T(0),
        T(0), T(0), T(0),
        T(0), T(0), T(0));
  }
};

template <class T>
[[nodiscard]] CPL_ALWAYS_INLINE constexpr
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
CPL_ALWAYS_INLINE constexpr
void operator+=(matrix3x3<T>& a, matrix3x3<T> const& b)
{
  a.xx() += b.xx();
  a.xy() += b.xy();
  a.xz() += b.xz();
  a.yx() += b.yx();
  a.yy() += b.yy();
  a.yz() += b.yz();
  a.zx() += b.zx();
  a.zy() += b.zy();
  a.zz() += b.zz();
}

template <class A, class B>
[[nodiscard]] CPL_ALWAYS_INLINE constexpr
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
[[nodiscard]] CPL_ALWAYS_INLINE constexpr
auto operator*(matrix3x3<A> const& a, vector3<B> const& b)
{
  using result_type = decltype(a.xx() * b.x());
  return vector3<result_type>(
      a.xx() * b.x() + a.xy() * b.y() + a.xz() * b.z(),
      a.yx() * b.x() + a.yy() * b.y() + a.yz() * b.z(),
      a.zx() * b.x() + a.zy() * b.y() + a.zz() * b.z());
}

template <class A, class B>
[[nodiscard]] CPL_ALWAYS_INLINE constexpr
auto outer_product(vector3<A> const& a, vector3<B> const& b)
{
  using result_type = decltype(a.x() * b.x());
  return matrix3x3<result_type>(
      a.x() * b.x(), a.x() * b.y(), a.x() * b.z(),
      a.y() * b.x(), a.y() * b.y(), a.y() * b.z(),
      a.z() * b.x(), a.z() * b.y(), a.z() * b.z());
}

template <class T>
[[nodiscard]] CPL_ALWAYS_INLINE constexpr
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
[[nodiscard]] CPL_ALWAYS_INLINE constexpr
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

}
