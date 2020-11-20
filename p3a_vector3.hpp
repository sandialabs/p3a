#pragma once

#include <type_traits>

#include "p3a_macros.hpp"
#include "p3a_scalar.hpp"

namespace p3a {

template <class T>
class vector3 {
  T m_x;
  T m_y;
  T m_z;
 public:
  using reference = T&;
  using const_reference = T const&;
  P3A_ALWAYS_INLINE vector3() = default;
  P3A_ALWAYS_INLINE constexpr vector3(T const& a, T const& b, T const& c)
    :m_x(a)
    ,m_y(b)
    ,m_z(c)
  {}
  template <class U>
  P3A_ALWAYS_INLINE explicit constexpr vector3(vector3<U> const& other)
    :vector3(T(other.x()), T(other.y()), T(other.z()))
  {
  }
  P3A_ALWAYS_INLINE constexpr void store(T* ptr, int stride, int index) const
  {
    ptr[stride * 0 + index] = m_x;
    ptr[stride * 1 + index] = m_y;
    ptr[stride * 2 + index] = m_z;
  }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr reference x() { return m_x; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr reference y() { return m_y; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr reference z() { return m_z; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr const_reference x() const { return m_x; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr const_reference y() const { return m_y; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr const_reference z() const { return m_z; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr
  auto volume() const { return m_x * m_y * m_z; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr
  T const& operator[](int pos) const
  {
    if (pos == 0) return m_x;
    if (pos == 1) return m_y;
    return m_z;
  }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr
  T& operator[](int pos)
  {
    if (pos == 0) return m_x;
    if (pos == 1) return m_y;
    return m_z;
  }
  [[nodiscard]] P3A_ALWAYS_INLINE static constexpr
  vector3 zero()
  {
    return vector3(T(0), T(0), T(0));
  }
  [[nodiscard]] P3A_ALWAYS_INLINE static constexpr
  vector3 ones()
  {
    return vector3(T(1), T(1), T(1));
  }
  [[nodiscard]] P3A_ALWAYS_INLINE static constexpr
  vector3 load(T const* ptr, int stride, int index)
  {
    return vector3(
      ptr[stride * 0 + index],
      ptr[stride * 1 + index],
      ptr[stride * 2 + index]);
  }
};

template <class T>
[[nodiscard]] P3A_ALWAYS_INLINE constexpr
bool operator==(vector3<T> const& a, vector3<T> const& b)
{
  return a.x() == b.x() &&
         a.y() == b.y() &&
         a.z() == b.z();
}

template <class A, class B>
P3A_ALWAYS_INLINE constexpr
void operator+=(vector3<A>& a, vector3<B> const& b)
{
  a.x() += b.x();
  a.y() += b.y();
  a.z() += b.z();
}

template <class A, class B>
P3A_ALWAYS_INLINE constexpr
void operator-=(vector3<A>& a, vector3<B> const& b)
{
  a.x() -= b.x();
  a.y() -= b.y();
  a.z() -= b.z();
}

template <class A, class B>
P3A_ALWAYS_INLINE constexpr
void operator*=(vector3<A>& a, B const& b)
{
  a.x() *= b;
  a.y() *= b;
  a.z() *= b;
}

template <class A, class B>
P3A_ALWAYS_INLINE constexpr
void operator/=(vector3<A>& a, B const& b)
{
  a.x() /= b;
  a.y() /= b;
  a.z() /= b;
}

template <class A, class B>
[[nodiscard]] P3A_ALWAYS_INLINE constexpr
auto operator+(vector3<A> const& a, vector3<B> const& b) {
  using C = decltype(a.x() + b.x());
  return vector3<C>(a.x() + b.x(), a.y() + b.y(), a.z() + b.z());
}

template <class A, class B>
[[nodiscard]] P3A_ALWAYS_INLINE constexpr
auto operator-(vector3<A> const& a, vector3<B> const& b) {
  using C = decltype(a.x() - b.x());
  return vector3<C>(a.x() - b.x(), a.y() - b.y(), a.z() - b.z());
}

template <class A, class B>
[[nodiscard]] P3A_ALWAYS_INLINE constexpr
auto operator/(vector3<A> const& a, B const& b) {
  using C = decltype(a.x() / b);
  return vector3<C>(a.x() / b, a.y() / b, a.z() / b);
}

template <class A, class B>
[[nodiscard]] P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, vector3<decltype(A() * B())>>::type
operator*(vector3<A> const& a, B const& b) {
  using C = decltype(a.x() * b);
  return vector3<C>(a.x() * b, a.y() * b, a.z() * b);
}

template <class A, class B>
[[nodiscard]] P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<A>, vector3<decltype(A() * B())>>::type
operator*(A const& a, vector3<B> const& b) {
  return b * a;
}

template <class A>
[[nodiscard]] P3A_ALWAYS_INLINE constexpr
vector3<A> operator-(vector3<A> const& a) {
  return vector3<A>(-a.x(), -a.y(), -a.z());
}

template <class A, class B>
[[nodiscard]] P3A_ALWAYS_INLINE constexpr
auto hadamard_product(vector3<A> const& a, vector3<B> const& b) {
  using C = decltype(a.x() * b.x());
  return vector3<C>(a.x() * b.x(), a.y() * b.y(), a.z() * b.z());
}

template <class A, class B>
[[nodiscard]] P3A_ALWAYS_INLINE constexpr
auto hadamard_division(vector3<A> const& a, vector3<B> const& b) {
  using C = decltype(a.x() / b.x());
  return vector3<C>(a.x() / b.x(), a.y() / b.y(), a.z() / b.z());
}

template <class A, class B>
[[nodiscard]] P3A_ALWAYS_INLINE constexpr
auto hadamard_equality(vector3<A> const& a, vector3<B> const& b) {
  using C = decltype(a.x() == b.x());
  return vector3<C>(a.x() == b.x(), a.y() == b.y(), a.z() == b.z());
}

template <class A, class B>
[[nodiscard]] P3A_ALWAYS_INLINE constexpr
vector3<B> hadamard_condition(
    vector3<A> const& a,
    vector3<B> const& b,
    vector3<B> const& c) {
  return vector3<B>(
      condition(a.x(), b.x(), c.x()),
      condition(a.y(), b.y(), c.y()),
      condition(a.z(), b.z(), c.z()));
}

template <class A, class B>
[[nodiscard]] P3A_ALWAYS_INLINE constexpr
auto hadamard_disjunction(vector3<A> const& a, vector3<B> const& b) {
  using C = decltype(a.x() || b.x());
  return vector3<C>(a.x() || b.x(), a.y() || b.y(), a.z() || b.z());
}

template <class A, class B>
[[nodiscard]] P3A_ALWAYS_INLINE constexpr
auto cross_product(vector3<A> const& a, vector3<B> const& b) {
  using C = decltype(a.x() * b.y());
  return vector3<C>(
      a.y() * b.z() - a.z() * b.y(),
      a.z() * b.x() - a.x() * b.z(),
      a.x() * b.y() - a.y() * b.x());
}

template <class A, class B>
[[nodiscard]] P3A_ALWAYS_INLINE constexpr
auto dot_product(vector3<A> const& a, vector3<B> const& b) {
  return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}

template <class A, class B, class C>
[[nodiscard]] P3A_ALWAYS_INLINE constexpr
auto scalar_triple_product(
    vector3<A> const& a,
    vector3<B> const& b,
    vector3<C> const& c) {
  return dot_product(a, cross_product(b, c));
}

template <class T>
[[nodiscard]] P3A_ALWAYS_INLINE constexpr
T length(vector3<T> const& a) {
  return square_root(dot_product(a, a));
}

template <class T>
[[nodiscard]] P3A_ALWAYS_INLINE constexpr
auto normalize(vector3<T> const& a) {
  return a / length(a);
}

}
