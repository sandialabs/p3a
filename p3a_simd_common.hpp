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

class element_aligned_tag {};

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
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, Abi> operator+(T const& a, simd<T, Abi> const& b) {
  return simd<T, Abi>(a) + b;
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, Abi> operator+(simd<T, Abi> const& a, T const& b) {
  return a + simd<T, Abi>(b);
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, Abi> operator-(T const& a, simd<T, Abi> const& b) {
  return simd<T, Abi>(a) - b;
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, Abi> operator-(simd<T, Abi> const& a, T const& b) {
  return a - simd<T, Abi>(b);
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, Abi> operator*(T const& a, simd<T, Abi> const& b) {
  return simd<T, Abi>(a) * b;
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, Abi> operator*(simd<T, Abi> const& a, T const& b) {
  return a * simd<T, Abi>(b);
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, Abi> operator/(T const& a, simd<T, Abi> const& b) {
  return simd<T, Abi>(a) / b;
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, Abi> operator/(simd<T, Abi> const& a, T const& b) {
  return a / simd<T, Abi>(b);
}

P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr bool
all_of(bool a) { return a; }

P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr bool
any_of(bool a) { return a; }

}
