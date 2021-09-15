#pragma once

#include <cmath>
#include <cstdint>

#include "p3a_macros.hpp"
#include "p3a_functions.hpp"

namespace p3a {

template <class T, class Abi>
class simd;

template <class T, class Abi>
class simd_mask;

template <class T, class Abi>
class simd_index;

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, Abi>& operator+=(simd<T, Abi>& a, simd<T, Abi> const& b) {
  a = a + b;
  return a;
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, Abi>& operator-=(simd<T, Abi>& a, simd<T, Abi> const& b) {
  a = a - b;
  return a;
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, Abi>& operator*=(simd<T, Abi>& a, simd<T, Abi> const& b) {
  a = a * b;
  return a;
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, Abi>& operator/=(simd<T, Abi>& a, simd<T, Abi> const& b) {
  a = a / b;
  return a;
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_index<T, Abi>& operator+=(simd_index<T, Abi>& a, simd_index<T, Abi> const& b) {
  a = a + b;
  return a;
}

template <class U, class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
std::enable_if_t<std::is_arithmetic_v<U>, simd<T, Abi>>
operator+(U const& a, simd<T, Abi> const& b) {
  return simd<T, Abi>(T(a)) + b;
}

template <class U, class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
std::enable_if_t<std::is_arithmetic_v<U>, simd<T, Abi>>
operator+(simd<T, Abi> const& a, U const& b) {
  return a + simd<T, Abi>(T(b));
}

template <class U, class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
std::enable_if_t<std::is_arithmetic_v<U>, simd<T, Abi>>
operator-(U const& a, simd<T, Abi> const& b) {
  return simd<T, Abi>(T(a)) - b;
}

template <class U, class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
std::enable_if_t<std::is_arithmetic_v<U>, simd<T, Abi>>
operator-(simd<T, Abi> const& a, U const& b) {
  return a - simd<T, Abi>(T(b));
}

template <class U, class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
std::enable_if_t<std::is_arithmetic_v<U>, simd<T, Abi>>
operator*(U const& a, simd<T, Abi> const& b) {
  return simd<T, Abi>(T(a)) * b;
}

template <class U, class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
std::enable_if_t<std::is_arithmetic_v<U>, simd<T, Abi>>
operator*(simd<T, Abi> const& a, U const& b) {
  return a * simd<T, Abi>(T(b));
}

template <class U, class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
std::enable_if_t<std::is_arithmetic_v<U>, simd<T, Abi>>
operator/(U const& a, simd<T, Abi> const& b) {
  return simd<T, Abi>(T(a)) / b;
}

template <class U, class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
std::enable_if_t<std::is_arithmetic_v<U>, simd<T, Abi>>
operator/(simd<T, Abi> const& a, U const& b) {
  return a / simd<T, Abi>(T(b));
}

template <class U, class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
std::enable_if_t<std::is_arithmetic_v<U>, simd<T, Abi>&>
operator/=(simd<T, Abi>& a, U const& b) {
  a = a / b;
  return a;
}

P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr bool
all_of(bool a) { return a; }

P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr bool
any_of(bool a) { return a; }

namespace details {

template <class T, class Abi>
struct is_scalar<simd<T, Abi>> {
  inline static constexpr bool value = true;
};

}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
simd<T, Abi> load(
    T const* ptr, int offset, simd_mask<T, Abi> const& mask)
{
  return simd<T, Abi>::masked_load(ptr + offset, mask);
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
simd<T, Abi> load(
    T const* ptr, simd_index<T, Abi> const& offset, simd_mask<T, Abi> const& mask)
{
  return simd<T, Abi>::masked_gather(ptr, offset, mask);
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
void store(
    simd<T, Abi> const& value,
    T* ptr, int offset, simd_mask<T, Abi> const& mask)
{
  return value.masked_store(ptr + offset, mask);
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
void store(
    simd<T, Abi> const& value,
    T* ptr, simd_index<T, Abi> const& offset, simd_mask<T, Abi> const& mask)
{
  return value.masked_scatter(ptr, offset, mask);
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
T get(simd<T, Abi> const& value, int i)
{
  T storage[simd<T, Abi>::size()];
  value.store(storage);
  return storage[i];
}

}
