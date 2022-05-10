#pragma once

#include <type_traits>

#include "p3a_macros.hpp"
#include "p3a_scalar.hpp"
#include "p3a_functions.hpp"

namespace p3a {

template <class T>
class vector2 {
  T m_x;
  T m_y;
 public:
  using reference = T&;
  using const_reference = T const&;
  P3A_ALWAYS_INLINE vector2() = default;
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  vector2(T const& a, T const& b)
    :m_x(a)
    ,m_y(b)
  {}
  template <class U>
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE explicit constexpr
  vector2(vector2<U> const& other)
    :vector2(T(other.x()), T(other.y()))
  {
  }
  template <
    class U,
    class V,
    typename std::enable_if<
      std::is_constructible_v<T, U const&> &&
      std::is_constructible_v<T, V const&>,
      bool>::type = false>
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE explicit constexpr
  vector2(U const& a, V const& b)
    :m_x(a)
    ,m_y(b)
  {}
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  reference x() { return m_x; }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  reference y() { return m_y; }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  const_reference x() const { return m_x; }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  const_reference y() const { return m_y; }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  auto area() const { return m_x * m_y; }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& operator[](int pos) const
  {
    if (pos == 0) return m_x;
    return m_y;
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  T& operator[](int pos)
  {
    if (pos == 0) return m_x;
    return m_y;
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE static constexpr
  vector2 zero()
  {
    return vector2(T(0), T(0));
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE static constexpr
  vector2 ones()
  {
    return vector2(T(1), T(1));
  }
  template <class U>
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  vector2& operator=(vector2<U> const& other)
  {
    m_x = other.x();
    m_y = other.y();
    return *this;
  }
};

template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
bool operator==(vector2<T> const& a, vector2<T> const& b)
{
  return a.x() == b.x() &&
         a.y() == b.y();
}

template <class A, class B>
P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
void operator+=(vector2<A>& a, vector2<B> const& b)
{
  a.x() += b.x();
  a.y() += b.y();
}

template <class A, class B>
P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
void operator-=(vector2<A>& a, vector2<B> const& b)
{
  a.x() -= b.x();
  a.y() -= b.y();
}

template <class A, class B>
P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
void operator*=(vector2<A>& a, B const& b)
{
  a.x() *= b;
  a.y() *= b;
}

template <class A, class B>
P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
void operator/=(vector2<A>& a, B const& b)
{
  a.x() /= b;
  a.y() /= b;
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator+(vector2<A> const& a, vector2<B> const& b) {
  using C = decltype(a.x() + b.x());
  return vector2<C>(a.x() + b.x(), a.y() + b.y());
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator-(vector2<A> const& a, vector2<B> const& b) {
  using C = decltype(a.x() - b.x());
  return vector2<C>(a.x() - b.x(), a.y() - b.y());
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator/(vector2<A> const& a, B const& b) {
  using C = decltype(a.x() / b);
  return vector2<C>(a.x() / b, a.y() / b);
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
typename std::enable_if<is_scalar<B>, vector2<decltype(A() * B())>>::type
operator*(vector2<A> const& a, B const& b) {
  using C = decltype(a.x() * b);
  return vector2<C>(a.x() * b, a.y() * b);
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<A>, vector2<decltype(A() * B())>>::type
operator*(A const& a, vector2<B> const& b) {
  return b * a;
}

template <class A>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
vector2<A> operator-(vector2<A> const& a) {
  return vector2<A>(-a.x(), -a.y());
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto hadamard_product(vector2<A> const& a, vector2<B> const& b) {
  using C = decltype(a.x() * b.x());
  return vector2<C>(a.x() * b.x(), a.y() * b.y());
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto hadamard_division(vector2<A> const& a, vector2<B> const& b) {
  using C = decltype(a.x() / b.x());
  return vector2<C>(a.x() / b.x(), a.y() / b.y());
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto hadamard_equality(vector2<A> const& a, vector2<B> const& b) {
  using C = decltype(a.x() == b.x());
  return vector2<C>(a.x() == b.x(), a.y() == b.y());
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
vector2<B> hadamard_condition(
    vector2<A> const& a,
    vector2<B> const& b,
    vector2<B> const& c) {
  return vector2<B>(
      condition(a.x(), b.x(), c.x()),
      condition(a.y(), b.y(), c.y()));
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto hadamard_disjunction(vector2<A> const& a, vector2<B> const& b) {
  using C = decltype(a.x() || b.x());
  return vector2<C>(a.x() || b.x(), a.y() || b.y());
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto cross_product(vector2<A> const& a, vector2<B> const& b) {
  return a.x() * b.y() - a.y() * b.x();
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto dot_product(vector2<A> const& a, vector2<B> const& b) {
  return a.x() * b.x() + a.y() * b.y();
}

template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
T magnitude(vector2<T> const& a) {
  return square_root(dot_product(a, a));
}

template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto normalize(vector2<T> const& a) {
  return a / magnitude(a);
}

template <class T>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
vector2<T> load_vector2(
    T const* ptr, int stride, int offset)
{
  return vector2<T>(
      load(ptr, 0 * stride + offset),
      load(ptr, 1 * stride + offset));
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
void store(
    vector2<T> const& value,
    T* ptr, int stride, int offset)
{
  store(value.x(), ptr, 0 * stride + offset);
  store(value.y(), ptr, 1 * stride + offset);
}

}
