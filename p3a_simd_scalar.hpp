#pragma once

#include "p3a_simd_common.hpp"
#include "p3a_type_traits.hpp"

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
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE static constexpr int size() { return 1; }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd_mask(value_type value)
    :m_value(value)
  {}
  template <class U>
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
  simd_mask(simd_mask<U, simd_abi::scalar> const& other)
    :m_value(other.get())
  {
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline constexpr bool get() const { return m_value; }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd_mask operator||(simd_mask const& other) const {
    return m_value || other.m_value;
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd_mask operator&&(simd_mask const& other) const {
    return m_value && other.m_value;
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd_mask operator!() const {
    return !m_value;
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline bool operator==(simd_mask const& other) const {
    return m_value == other.m_value;
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE static inline
  simd_mask first_n(int n)
  {
    return simd_mask(n != 0);
  }
};

template <class T>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline bool none_of(simd_mask<T, simd_abi::scalar> const& mask) {
  return !mask.get();
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
bool all_of(simd_mask<T, simd_abi::scalar> const& a)
{ return a.get(); }

template <class T>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
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
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE static constexpr int size() { return 1; }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd(value_type value)
    :m_value(value)
  {}
  template <class U>
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline explicit simd(simd<U, abi_type> const& other)
    :m_value(value_type(other.get()))
  {}
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd operator*(simd const& other) const {
    return simd(m_value * other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd operator/(simd const& other) const {
    return simd(m_value / other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd operator+(simd const& other) const {
    return simd(m_value + other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd operator-(simd const& other) const {
    return simd(m_value - other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd operator-() const {
    return simd(-m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd operator>>(unsigned int rhs) const {
    return simd(m_value >> rhs);
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd operator>>(simd<std::uint32_t, abi_type> const& rhs) const {
    return simd(m_value >> rhs.get());
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd operator<<(unsigned int rhs) const {
    return simd(m_value << rhs);
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd operator<<(simd<std::uint32_t, abi_type> const& rhs) const {
    return simd(m_value << rhs.get());
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd operator&(simd const& other) const {
    return m_value & other.get();
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd operator|(simd const& other) const {
    return m_value | other.get();
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE constexpr T get() const { return m_value; }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline mask_type operator<(simd const& other) const {
    return mask_type(m_value < other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline mask_type operator>(simd const& other) const {
    return mask_type(m_value > other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline mask_type operator<=(simd const& other) const {
    return mask_type(m_value <= other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline mask_type operator>=(simd const& other) const {
    return mask_type(m_value >= other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline mask_type operator==(simd const& other) const {
    return mask_type(m_value == other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline mask_type operator!=(simd const& other) const {
    return mask_type(m_value != other.m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline void copy_from(T const* ptr, element_aligned_tag) {
    m_value = *ptr;
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline void copy_to(T* ptr, element_aligned_tag) const {
    *ptr = m_value;
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE static inline simd zero() {
    return simd(T(0));
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE static inline simd contiguous_from(value_type i) {
    return simd(i);
  }
};

template <class T>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
simd<T, simd_abi::scalar> abs(simd<T, simd_abi::scalar> const& a)
{
  return simd<T, simd_abi::scalar>(std::abs(a.get()));
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd<T, simd_abi::scalar> square_root(simd<T, simd_abi::scalar> const& a) {
  return simd<T, simd_abi::scalar>(std::sqrt(a.get()));
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd<T, simd_abi::scalar> fma(
    simd<T, simd_abi::scalar> const& a,
    simd<T, simd_abi::scalar> const& b,
    simd<T, simd_abi::scalar> const& c) {
  return simd<T, simd_abi::scalar>((a.get() * b.get()) + c.get());
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd<T, simd_abi::scalar> max(
    simd<T, simd_abi::scalar> const& a, simd<T, simd_abi::scalar> const& b) {
  return simd<T, simd_abi::scalar>(condition((a.get() < b.get()), b.get(), a.get()));
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd<T, simd_abi::scalar> min(
    simd<T, simd_abi::scalar> const& a, simd<T, simd_abi::scalar> const& b) {
  return simd<T, simd_abi::scalar>(condition((b.get() < a.get()), b.get(), a.get()));
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd<T, simd_abi::scalar>
condition(
    no_deduce_t<simd_mask<T, simd_abi::scalar>> const& a,
    simd<T, simd_abi::scalar> const& b,
    simd<T, simd_abi::scalar> const& c)
{
  return simd<T, simd_abi::scalar>(condition(a.get(), b.get(), c.get()));
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline simd<T, Abi>
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
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
    :m_value(const_cast<value_type&>(value_arg))
    ,m_mask(mask_arg)
  {}
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
  mask_type const& mask() const { return m_mask; }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
  value_type const& value() const { return m_value; }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
  void copy_to(T* mem, element_aligned_tag) const {
    if (m_mask.get()) *mem = m_value.get();
  }
  template <class Integral>
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
  std::enable_if_t<std::is_integral_v<Integral>, void>
  scatter_to(T* mem, simd<Integral, simd_abi::scalar> const& index) const {
    if (m_mask.get()) mem[index.get()] = m_value.get();
  }
};

template <class T>
class where_expression<simd_mask<T, simd_abi::scalar>, simd<T, simd_abi::scalar>>
 : public const_where_expression<simd_mask<T, simd_abi::scalar>, simd<T, simd_abi::scalar>> {
  using base_type = const_where_expression<simd_mask<T, simd_abi::scalar>, simd<T, simd_abi::scalar>>;
 public:
  using typename base_type::value_type;
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
  where_expression(simd_mask<T, simd_abi::scalar> const& mask_arg, simd<T, simd_abi::scalar>& value_arg)
    :base_type(mask_arg, value_arg)
  {}
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
  void copy_from(T const* mem, element_aligned_tag) {
    this->m_value = value_type(this->m_mask.get() ? *mem : T(0));
  }
  template <class Integral>
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
  std::enable_if_t<std::is_integral_v<Integral>, void>
  gather_from(T const* mem, simd<Integral, simd_abi::scalar> const& index) {
    this->m_value = value_type(this->m_mask.get() ? mem[index.get()] : T(0));
  }
};

template <class T, class BinaryOp>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
T reduce(
    const_where_expression<simd_mask<T, simd_abi::scalar>, simd<T, simd_abi::scalar>> const& x,
    T identity_element,
    BinaryOp)
{
  return x.mask().get() ? x.value().get() : identity_element;
}

template <class To, class From>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline constexpr
To bit_cast(simd<From, simd_abi::scalar> const& src)
{
  return To(p3a::bit_cast<typename To::value_type>(src.get()));
}

}
