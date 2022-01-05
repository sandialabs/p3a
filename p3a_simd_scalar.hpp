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
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE static inline
  simd_mask first_n(int n)
  {
    return simd_mask(n != 0);
  }
};

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline bool none_of(simd_mask<T, simd_abi::scalar> const& mask) {
  return !mask.get();
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
bool all_of(simd_mask<T, simd_abi::scalar> const& a)
{ return a.get(); }

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
bool any_of(simd_mask<T, simd_abi::scalar> const& a)
{ return a.get(); }

template <class T>
class simd_index<T, simd_abi::scalar> {
  int m_value;
 public:
  using value_type = int;
  using abi_type = simd_abi::scalar;
  P3A_ALWAYS_INLINE inline simd_index() = default;
  P3A_ALWAYS_INLINE inline simd_index(simd_index const&) = default;
  P3A_ALWAYS_INLINE inline simd_index(simd_index&&) = default;
  P3A_ALWAYS_INLINE inline simd_index& operator=(simd_index const&) = default;
  P3A_ALWAYS_INLINE inline simd_index& operator=(simd_index&&) = default;
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE static constexpr int size() { return 1; }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_index(int value)
    :m_value(value)
  {}
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_index operator*(simd_index const& other) const {
    return simd_index(m_value * other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_index operator/(simd_index const& other) const {
    return simd_index(m_value / other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_index operator+(simd_index const& other) const {
    return simd_index(m_value + other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_index operator-(simd_index const& other) const {
    return simd_index(m_value - other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_index operator-() const {
    return simd_index(-m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE constexpr int get() const { return m_value; }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_mask<T, simd_abi::scalar> operator<(simd_index const& other) const {
    return simd_mask<T, simd_abi::scalar>(m_value < other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_mask<T, simd_abi::scalar> operator>(simd_index const& other) const {
    return simd_mask<T, simd_abi::scalar>(m_value > other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_mask<T, simd_abi::scalar> operator<=(simd_index const& other) const {
    return simd_mask<T, simd_abi::scalar>(m_value <= other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_mask<T, simd_abi::scalar> operator>=(simd_index const& other) const {
    return simd_mask<T, simd_abi::scalar>(m_value >= other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_mask<T, simd_abi::scalar> operator==(simd_index const& other) const {
    return simd_mask<T, simd_abi::scalar>(m_value == other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_mask<T, simd_abi::scalar> operator!=(simd_index const& other) const {
    return simd_mask<T, simd_abi::scalar>(m_value != other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE static inline simd_index contiguous_from(int i) {
    return simd_index(i);
  }
};

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_index<T, simd_abi::scalar>
condition(
    simd_mask<T, simd_abi::scalar> const& a,
    simd_index<T, simd_abi::scalar> const& b,
    simd_index<T, simd_abi::scalar> const& c)
{
  return simd_index<T, simd_abi::scalar>(condition(a.get(), b.get(), c.get()));
}

template <class T>
class simd<T, simd_abi::scalar> {
  T m_value;
 public:
  using value_type = T;
  using abi_type = simd_abi::scalar;
  using mask_type = simd_mask<T, abi_type>;
  using index_type = simd_index<T, abi_type>;
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
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_mask<T, simd_abi::scalar> operator>(simd const& other) const {
    return simd_mask<T, simd_abi::scalar>(m_value > other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_mask<T, simd_abi::scalar> operator<=(simd const& other) const {
    return simd_mask<T, simd_abi::scalar>(m_value <= other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_mask<T, simd_abi::scalar> operator>=(simd const& other) const {
    return simd_mask<T, simd_abi::scalar>(m_value >= other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_mask<T, simd_abi::scalar> operator==(simd const& other) const {
    return simd_mask<T, simd_abi::scalar>(m_value == other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd_mask<T, simd_abi::scalar> operator!=(simd const& other) const {
    return simd_mask<T, simd_abi::scalar>(m_value != other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline void copy_from(T const* ptr, element_aligned_tag) {
    m_value = *ptr;
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline void copy_to(T* ptr, element_aligned_tag) const {
    *ptr = m_value;
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE static inline simd zero() {
    return simd(T(0));
  }
};

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, simd_abi::scalar> absolute_value(simd<T, simd_abi::scalar> const& a) {
  return simd<T, simd_abi::scalar>(std::abs(a.get()));
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, simd_abi::scalar> square_root(simd<T, simd_abi::scalar> const& a) {
  return simd<T, simd_abi::scalar>(std::sqrt(a.get()));
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
  return simd<T, simd_abi::scalar>(condition((a.get() < b.get()), b.get(), a.get()));
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, simd_abi::scalar> min(
    simd<T, simd_abi::scalar> const& a, simd<T, simd_abi::scalar> const& b) {
  return simd<T, simd_abi::scalar>(condition((b.get() < a.get()), b.get(), a.get()));
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, simd_abi::scalar>
condition(
    simd_mask<T, simd_abi::scalar> const& a,
    simd<T, simd_abi::scalar> const& b,
    simd<T, simd_abi::scalar> const& c)
{
  return simd<T, simd_abi::scalar>(condition(a.get(), b.get(), c.get()));
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline simd<T, Abi>
copysign(simd<T, Abi> a, simd<T, Abi> b) {
  return std::copysign(a.get(), b.get());
}

template <class T>
class const_where_expression<simd_mask<T, simd_abi::scalar>, simd<T, simd_abi::scalar>> {
 public:
  using abi_type = simd_abi::scalar;
  using value_type = simd<T, abi_type>;
  using mask_type = simd_mask<T, abi_type>;
 protected:
  value_type& m_value;
  mask_type const& m_mask;
 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
    :m_value(const_cast<value_type&>(value_arg))
    ,m_mask(mask_arg)
  {}
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  void copy_to(T* mem, element_aligned_tag) const {
    if (m_mask.get()) *mem = m_value.get();
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  void scatter_to(T* mem, simd_index<T, simd_abi::scalar> const& index) const {
    if (m_mask.get()) mem[index.get()] = m_value.get();
  }
};

template <class T>
class where_expression<simd_mask<T, simd_abi::scalar>, simd<T, simd_abi::scalar>>
 : public const_where_expression<simd_mask<T, simd_abi::scalar>, simd<T, simd_abi::scalar>> {
  using base_type = const_where_expression<simd_mask<T, simd_abi::scalar>, simd<T, simd_abi::scalar>>;
 public:
  using typename base_type::value_type;
  where_expression(simd_mask<T, simd_abi::scalar> const& mask_arg, simd<T, simd_abi::scalar>& value_arg)
    :base_type(mask_arg, value_arg)
  {}
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  void copy_from(T const* mem, element_aligned_tag) {
    this->m_value = value_type(this->m_mask.get() ? *mem : T(0));
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  void gather_from(T const* mem, simd_index<T, simd_abi::scalar> const& index) {
    this->m_value = value_type(this->m_mask.get() ? mem[index.get()] : T(0));
  }
};

}
