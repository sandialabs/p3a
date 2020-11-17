#pragma once

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
  CPL_ALWAYS_INLINE constexpr symmetric3x3() = default;
  CPL_ALWAYS_INLINE constexpr
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
  CPL_ALWAYS_INLINE constexpr
  symmetric3x3(scaled_identity3x3<T> const& a)
    :symmetric3x3(a.scale(), T(0), T(0), a.scale(), T(0), a.scale())
  {}
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T const& xx() const { return m_xx; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T const& xy() const { return m_xy; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T const& xz() const { return m_xz; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T const& yx() const { return m_xy; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T const& yy() const { return m_yy; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T const& yz() const { return m_yz; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T const& zx() const { return m_xz; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T const& zy() const { return m_yz; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T const& zz() const { return m_zz; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T& xx() { return m_xx; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T& xy() { return m_xy; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T& xz() { return m_xz; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T& yx() { return m_xy; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T& yy() { return m_yy; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T& yz() { return m_yz; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T& zx() { return m_xz; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T& zy() { return m_yz; }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  T& zz() { return m_zz; }
  [[nodiscard]] CPL_ALWAYS_INLINE static constexpr
  symmetric3x3<T> zero()
  {
    return symmetric3x3<T>(
        T(0), T(0), T(0),
        T(0), T(0), T(0));
  }
  [[nodiscard]] CPL_ALWAYS_INLINE static constexpr
  symmetric3x3<T> load(T const* ptr, int stride, int index)
  {
    return symmetric3x3<T>(
        ptr[stride * 0 + index],
        ptr[stride * 1 + index],
        ptr[stride * 2 + index],
        ptr[stride * 3 + index],
        ptr[stride * 4 + index],
        ptr[stride * 5 + index]);
  }
  CPL_ALWAYS_INLINE constexpr
  void store(T* ptr, int stride, int index) const
  {
    ptr[stride * 0 + index] = m_xx;
    ptr[stride * 1 + index] = m_xy;
    ptr[stride * 2 + index] = m_xz;
    ptr[stride * 3 + index] = m_yy;
    ptr[stride * 4 + index] = m_yz;
    ptr[stride * 5 + index] = m_zz;
  }
};

template <class T>
CPL_ALWAYS_INLINE constexpr
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
[[nodiscard]] CPL_ALWAYS_INLINE constexpr
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
[[nodiscard]] CPL_ALWAYS_INLINE constexpr
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
[[nodiscard]] CPL_ALWAYS_INLINE constexpr
T trace(symmetric3x3<T> const& a)
{
  return a.xx() + a.yy() + a.zz();
}

template <class T>
[[nodiscard]] CPL_ALWAYS_INLINE constexpr
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
[[nodiscard]] CPL_ALWAYS_INLINE constexpr
symmetric3x3<T> operator+(
    scaled_identity3x3<T> const& a,
    symmetric3x3<T> const& b)
{
  return b + a;
}

template <class T>
[[nodiscard]] CPL_ALWAYS_INLINE constexpr
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
[[nodiscard]] CPL_ALWAYS_INLINE constexpr
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
[[nodiscard]] CPL_ALWAYS_INLINE constexpr
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
[[nodiscard]] CPL_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<A>, symmetric3x3<decltype(A() * B())>>::type
operator*(
    A const& a,
    symmetric3x3<B> const& b)
{
  return b * a;
}

template <class A, class B>
[[nodiscard]] CPL_ALWAYS_INLINE constexpr
auto operator*(symmetric3x3<A> const& a, vector3<B> const& b)
{
  using C = decltype(a.xx() * b.x());
  return vector3<C>(
      a.xx() * b.x() + a.xy() * b.y() + a.xz() * b.z(),
      a.yx() * b.x() + a.yy() * b.y() + a.yz() * b.z(),
      a.zx() * b.x() + a.zy() * b.y() + a.zz() * b.z());
}

template <class A, class B>
CPL_ALWAYS_INLINE constexpr
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

inline int constexpr symmetric3x3_component_count = 6;

}
