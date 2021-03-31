#pragma once

#include "p3a_simd_common.hpp"

namespace p3a {

namespace simd_abi {

class scalar {};

}

template <class T>
class simd_mask<T, simd_abi::scalar> {
  bool m_value;
 public:
  using value_type = bool;
  using simd_type = simd<T, simd_abi::scalar>;
  using abi_type = simd_abi::scalar;
  P3A_ALWAYS_INLINE inline simd_mask() = default;
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE static constexpr int size() { return 1; }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_mask(bool value)
    :m_value(value)
  {}
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr bool get() const { return m_value; }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_mask operator||(simd_mask const& other) const {
    return m_value || other.m_value;
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_mask operator&&(simd_mask const& other) const {
    return m_value && other.m_value;
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_mask operator!() const {
    return !m_value;
  }
};

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
bool all_of(simd_mask<T, simd_abi::scalar> const& a)
{ return a.get(); }

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
bool any_of(simd_mask<T, simd_abi::scalar> const& a)
{ return a.get(); }

template <class T>
class simd<T, simd_abi::scalar> {
  T m_value;
 public:
  using value_type = T;
  using abi_type = simd_abi::scalar;
  using mask_type = simd_mask<T, abi_type>;
  P3A_ALWAYS_INLINE inline simd() = default;
  P3A_ALWAYS_INLINE inline simd(simd const&) = default;
  P3A_ALWAYS_INLINE inline simd(simd&&) = default;
  P3A_ALWAYS_INLINE inline simd& operator=(simd const&) = default;
  P3A_ALWAYS_INLINE inline simd& operator=(simd&&) = default;
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE static constexpr int size() { return 1; }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd(T value)
    :m_value(value)
  {}
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd operator*(simd const& other) const {
    return simd(m_value * other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd operator/(simd const& other) const {
    return simd(m_value / other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd operator+(simd const& other) const {
    return simd(m_value + other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd operator-(simd const& other) const {
    return simd(m_value - other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd operator-() const {
    return simd(-m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE constexpr T get() const { return m_value; }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_mask<T, simd_abi::scalar> operator<(simd const& other) const {
    return simd_mask<T, simd_abi::scalar>(m_value < other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_mask<T, simd_abi::scalar> operator==(simd const& other) const {
    return simd_mask<T, simd_abi::scalar>(m_value == other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE static inline simd load(T const* ptr) {
    return simd(ptr[i]);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline void store(T* ptr) const {
    return ptr[i] = m_value;
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE static inline simd masked_load(T const* ptr, mask_type const& mask) {
    return simd(mask.get() ? ptr[i] : T(0));
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline void masked_store(T* ptr, mask_type const& mask) const {
    if (mask.get()) ptr[i] = m_value;
  }
};

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, simd_abi::scalar> abs(simd<T, simd_abi::scalar> const& a) {
  return simd<T, simd_abi::scalar>(std::abs(a.get()));
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, simd_abi::scalar> sqrt(simd<T, simd_abi::scalar> const& a) {
  return simd<T, simd_abi::scalar>(std::sqrt(a.get()));
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, simd_abi::scalar> cbrt(simd<T, simd_abi::scalar> const& a) {
  return simd<T, simd_abi::scalar>(std::cbrt(a.get()));
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, simd_abi::scalar> exp(simd<T, simd_abi::scalar> const& a) {
  return simd<T, simd_abi::scalar>(std::exp(a.get()));
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, simd_abi::scalar> fma(
    simd<T, simd_abi::scalar> const& a,
    simd<T, simd_abi::scalar> const& b,
    simd<T, simd_abi::scalar> const& c) {
  return simd<T, simd_abi::scalar>((a.get() * b.get()) + c.get());
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, simd_abi::scalar> max(
    simd<T, simd_abi::scalar> const& a, simd<T, simd_abi::scalar> const& b) {
  return simd<T, simd_abi::scalar>(choose((a.get() < b.get()), b.get(), a.get()));
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, simd_abi::scalar> min(
    simd<T, simd_abi::scalar> const& a, simd<T, simd_abi::scalar> const& b) {
  return simd<T, simd_abi::scalar>(choose((b.get() < a.get()), b.get(), a.get()));
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, simd_abi::scalar> choose(
    simd_mask<T, simd_abi::scalar> const& a, simd<T, simd_abi::scalar> const& b, simd<T, simd_abi::scalar> const& c) {
  return simd<T, simd_abi::scalar>(choose(a.get(), b.get(), c.get()));
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, Abi>
copysign(simd<T, Abi> a, simd<T, Abi> b) {
  return std::copysign(a.get(), b.get());
}

}
