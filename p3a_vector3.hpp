#pragma once

#include <type_traits>

#include "p3a_macros.hpp"
#include "p3a_scalar.hpp"
#include "p3a_functions.hpp"
#include "p3a_simd.hpp"

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
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  vector3(T const& a, T const& b, T const& c)
    :m_x(a)
    ,m_y(b)
    ,m_z(c)
  {}
  template <class U,
           std::enable_if_t<std::is_convertible_v<U, T>, bool> = false>
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  vector3(vector3<U> const& other)
    :vector3(T(other.x()), T(other.y()), T(other.z()))
  {
  }
  template <class U,
           std::enable_if_t<!std::is_convertible_v<U, T>, bool> = false>
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE explicit constexpr
  vector3(vector3<U> const& other)
    :vector3(T(other.x()), T(other.y()), T(other.z()))
  {
  }
  template <
    class U,
    class V,
    class W,
    typename std::enable_if<
      std::is_constructible_v<T, U const&> &&
      std::is_constructible_v<T, V const&> &&
      std::is_constructible_v<T, W const&>,
      bool>::type = false>
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE explicit constexpr
  vector3(U const& a, V const& b, W const& c)
    :m_x(a)
    ,m_y(b)
    ,m_z(c)
  {}
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  reference x() { return m_x; }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  reference y() { return m_y; }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  reference z() { return m_z; }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  const_reference x() const { return m_x; }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  const_reference y() const { return m_y; }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  const_reference z() const { return m_z; }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  auto volume() const { return m_x * m_y * m_z; }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& operator[](int pos) const
  {
    if (pos == 0) return m_x;
    if (pos == 1) return m_y;
    return m_z;
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  T& operator[](int pos)
  {
    if (pos == 0) return m_x;
    if (pos == 1) return m_y;
    return m_z;
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE static constexpr
  vector3 zero()
  {
    return vector3(T(0), T(0), T(0));
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE static constexpr
  vector3 ones()
  {
    return vector3(T(1), T(1), T(1));
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE static constexpr
  vector3 axis(int i)
  {
    if (i == 0) return vector3(T(1), T(0), T(0));
    if (i == 1) return vector3(T(0), T(1), T(0));
    return vector3(T(0), T(0), T(1));
  }
  template <class U>
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  vector3& operator=(vector3<U> const& other)
  {
    m_x = other.x();
    m_y = other.y();
    m_z = other.z();
    return *this;
  }
};

template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
bool operator==(vector3<T> const& a, vector3<T> const& b)
{
  return a.x() == b.x() &&
         a.y() == b.y() &&
         a.z() == b.z();
}

template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
bool operator!=(vector3<T> const& a, vector3<T> const& b)
{
  return a.x() != b.x() ||
         a.y() != b.y() ||
         a.z() != b.z();
}

template <class A, class B>
P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
void operator+=(vector3<A>& a, vector3<B> const& b)
{
  a.x() += b.x();
  a.y() += b.y();
  a.z() += b.z();
}

template <class A, class B>
P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
void operator-=(vector3<A>& a, vector3<B> const& b)
{
  a.x() -= b.x();
  a.y() -= b.y();
  a.z() -= b.z();
}

template <class A, class B>
P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
void operator*=(vector3<A>& a, B const& b)
{
  a.x() *= b;
  a.y() *= b;
  a.z() *= b;
}

template <class A, class B>
P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
void operator/=(vector3<A>& a, B const& b)
{
  a.x() /= b;
  a.y() /= b;
  a.z() /= b;
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator+(vector3<A> const& a, vector3<B> const& b) {
  using C = decltype(a.x() + b.x());
  return vector3<C>(a.x() + b.x(), a.y() + b.y(), a.z() + b.z());
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator-(vector3<A> const& a, vector3<B> const& b) {
  using C = decltype(a.x() - b.x());
  return vector3<C>(a.x() - b.x(), a.y() - b.y(), a.z() - b.z());
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator/(vector3<A> const& a, B const& b) {
  using C = decltype(a.x() / b);
  return vector3<C>(a.x() / b, a.y() / b, a.z() / b);
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
typename std::enable_if<is_scalar<B>, vector3<decltype(A() * B())>>::type
operator*(vector3<A> const& a, B const& b) {
  using C = decltype(a.x() * b);
  return vector3<C>(a.x() * b, a.y() * b, a.z() * b);
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<A>, vector3<decltype(A() * B())>>::type
operator*(A const& a, vector3<B> const& b) {
  return b * a;
}

template <class A>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
vector3<A> operator-(vector3<A> const& a) {
  return vector3<A>(-a.x(), -a.y(), -a.z());
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto hadamard_product(vector3<A> const& a, vector3<B> const& b) {
  using C = decltype(a.x() * b.x());
  return vector3<C>(a.x() * b.x(), a.y() * b.y(), a.z() * b.z());
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto hadamard_division(vector3<A> const& a, vector3<B> const& b) {
  using C = decltype(a.x() / b.x());
  return vector3<C>(a.x() / b.x(), a.y() / b.y(), a.z() / b.z());
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto hadamard_equality(vector3<A> const& a, vector3<B> const& b) {
  using C = decltype(a.x() == b.x());
  return vector3<C>(a.x() == b.x(), a.y() == b.y(), a.z() == b.z());
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
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
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto hadamard_disjunction(vector3<A> const& a, vector3<B> const& b) {
  using C = decltype(a.x() || b.x());
  return vector3<C>(a.x() || b.x(), a.y() || b.y(), a.z() || b.z());
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto cross_product(vector3<A> const& a, vector3<B> const& b) {
  using C = decltype(a.x() * b.y());
  return vector3<C>(
      a.y() * b.z() - a.z() * b.y(),
      a.z() * b.x() - a.x() * b.z(),
      a.x() * b.y() - a.y() * b.x());
}

template <class A, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto dot_product(vector3<A> const& a, vector3<B> const& b) {
  return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}

template <class A, class B, class C>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto scalar_triple_product(
    vector3<A> const& a,
    vector3<B> const& b,
    vector3<C> const& c) {
  return dot_product(a, cross_product(b, c));
}

template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
auto magnitude_squared(vector3<T> const& a) {
  return dot_product(a, a);
}

template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
T magnitude(vector3<T> const& a)
{
  return hypot(a.x(), a.y(), a.z());
}

template <class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto normalize(vector3<T> const& a) {
  return a / magnitude(a);
}

template <class T>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
vector3<T> load_vector3(
    T const* ptr, int stride, int offset)
{
  return vector3<T>(
      load(ptr, 0 * stride + offset),
      load(ptr, 1 * stride + offset),
      load(ptr, 2 * stride + offset));
}

template <class T, class U, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
auto load_vector3(
    T const* ptr, int stride, int offset, simd_mask<U, Abi> const& mask)
{
  auto const a = load(ptr + 0 * stride, offset, mask);
  auto const b = load(ptr + 1 * stride, offset, mask);
  auto const c = load(ptr + 2 * stride, offset, mask);
  using component_type = std::remove_const_t<decltype(a)>;
  return vector3<component_type>(a, b, c);
}

template <class T, class I, class U, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
auto load_vector3(
    T const* ptr, int stride, simd<I, Abi> const& offset, simd_mask<U, Abi> const& mask)
{
  auto const a = load(ptr + 0 * stride, offset, mask);
  auto const b = load(ptr + 1 * stride, offset, mask);
  auto const c = load(ptr + 2 * stride, offset, mask);
  using component_type = std::remove_const_t<decltype(a)>;
  return vector3<component_type>(a, b, c);
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
void store(
    vector3<T> const& value,
    T* ptr, int stride, int offset)
{
  store(value.x(), ptr + 0 * stride, offset);
  store(value.y(), ptr + 1 * stride, offset);
  store(value.z(), ptr + 2 * stride, offset);
}

template <class T, class U, class V, class Abi>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
void store(
    vector3<T> const& value,
    U* ptr,
    int stride,
    int offset,
    simd_mask<V, Abi> const& mask)
{
  store(value.x(), ptr + 0 * stride, offset, mask);
  store(value.y(), ptr + 1 * stride, offset, mask);
  store(value.z(), ptr + 2 * stride, offset, mask);
}

template <class T, class U, class V, class Integral, class Abi>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
void store(
    vector3<T> const& value,
    U* ptr,
    int stride,
    simd<Integral, Abi> const& offset,
    simd_mask<V, Abi> const& mask)
{
  store(value.x(), ptr + 0 * stride, offset, mask);
  store(value.y(), ptr + 1 * stride, offset, mask);
  store(value.z(), ptr + 2 * stride, offset, mask);
}

template <class T, class Mask>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
vector3<T> condition(
    Mask const& a,
    vector3<T> const& b,
    vector3<T> const& c)
{
  return vector3<T>(
      condition(a, b.x(), c.x()),
      condition(a, b.y(), c.y()),
      condition(a, b.z(), c.z()));
}

template <class T, class U>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
T scalar_projection(vector3<T> const& a, vector3<U> const& b)
{
  return dot_product(a, b);
}

template <class T, class U>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
vector3<T> project(vector3<T> const& a, vector3<U> const& b)
{
  return scalar_projection(a, b) * b;
}

template <class T, class U>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
vector3<T> reject(vector3<T> const& a, vector3<U> const& b)
{
  return a - project(a, b);
}

}
