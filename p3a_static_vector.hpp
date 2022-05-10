#pragma once

#include <type_traits>

#include "p3a_macros.hpp"
#include "p3a_scalar.hpp"

namespace p3a {

template <class T, int N>
class static_vector {
  T m_data[N];
 public:
  using reference = T&;
  using const_reference = T const&;
  P3A_ALWAYS_INLINE static_vector() = default;
  template <class U>
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE explicit constexpr
  static_vector(static_vector<U, N> const& other)
  {
    for (int i = 0; i < N; ++i) {
      m_data[i] = T(other.m_data[i]);
    }
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  reference operator[](int i)
  {
    return m_data[i];
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  const_reference operator[](int i) const
  {
    return m_data[i];
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE static
  static_vector zero()
  {
    static_vector r;
    for (int i = 0; i < N; ++i) {
      r.m_data[i] = T(0);
    }
    return r;
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE static
  static_vector ones()
  {
    static_vector r;
    for (int i = 0; i < N; ++i) {
      r.m_data[i] = T(1);
    }
    return r;
  }
  template <class U>
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  static_vector& operator=(static_vector<U, N> const& other)
  {
    for (int i = 0; i < N; ++i) {
      m_data[i] = other.m_data[i];
    }
    return *this;
  }
};

template <class T, int N>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
bool operator==(static_vector<T, N> const& a, static_vector<T, N> const& b)
{
  bool equal = true;
  for (int i = 0; i < N; ++i) {
    if (a[i] != b[i]) equal = false;
  }
  return equal;
}

template <class A, class B, int N>
P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
void operator+=(static_vector<A, N>& a, static_vector<B, N> const& b)
{
  for (int i = 0; i < N; ++i) {
    a[i] += b[i];
  }
}

template <class A, class B, int N>
P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
void operator-=(static_vector<A, N>& a, static_vector<B, N> const& b)
{
  for (int i = 0; i < N; ++i) {
    a[i] -= b[i];
  }
}

template <class A, class B, int N>
P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
void operator*=(static_vector<A, N>& a, B const& b)
{
  for (int i = 0; i < N; ++i) {
    a[i] *= b;
  }
}

template <class A, class B, int N>
P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
void operator/=(static_vector<A, N>& a, B const& b)
{
  for (int i = 0; i < N; ++i) {
    a[i] /= b;
  }
}

template <class A, class B, int N>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator+(static_vector<A, N> const& a, static_vector<B, N> const& b)
{
  using C = decltype(a[0] + b[0]);
  static_vector<C, N> r;
  for (int i = 0; i < N; ++i) {
    r[i] = a[i] + b[i];
  }
  return r;
}

template <class A, class B, int N>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator-(static_vector<A, N> const& a, static_vector<B, N> const& b)
{
  using C = decltype(a[0] - b[0]);
  static_vector<C, N> r;
  for (int i = 0; i < N; ++i) {
    r[i] = a[i] - b[i];
  }
  return r;
}

template <class A, class B, int N>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator/(static_vector<A, N> const& a, B const& b)
{
  using C = decltype(a[0] / b);
  static_vector<C, N> r;
  for (int i = 0 ; i < N; ++i) {
    r[i] = a[i] / b;
  }
  return r;
}

template <class A, class B, int N>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, static_vector<decltype(A() * B()), N>>::type
operator*(static_vector<A, N> const& a, B const& b)
{
  using C = decltype(a[0] * b);
  static_vector<C, N> r;
  for (int i = 0; i < N; ++i) {
    r[i] = a[i] * b;
  }
  return r;
}

template <class A, class B, int N>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<A>, static_vector<decltype(A() * B()), N>>::type
operator*(A const& a, static_vector<B, N> const& b)
{
  return b * a;
}

template <class A, int N>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
static_vector<A, N> operator-(static_vector<A, N> const& a)
{
  static_vector<A, N> r;
  for (int i = 0; i < N; ++i) {
    r[i] = -a[i];
  }
  return r;
}

}
