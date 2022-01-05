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

class element_aligned_tag {};

template <class Mask, class Value>
class const_where_expression {
 protected:
  Value& m_value;
  Mask const& m_mask;
 public:
  const_where_expression(Mask const& mask_arg, Value const& value_arg)
    :m_value(const_cast<Value&>(value_arg))
    ,m_mask(mask_arg)
  {}
};

template <class Mask, class Value>
class where_expression : public const_where_expression<Mask, Value> {
  using base_type = const_where_expression<Mask, Value>;
 public:
  where_expression(Mask const& mask_arg, Value& value_arg)
    :base_type(mask_arg, value_arg)
  {}
};

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
where_expression<simd_mask<T, Abi>, simd<T, Abi>>
where(simd_mask<T, Abi> const& mask, simd<T, Abi>& value) {
  return where_expression(mask, value);
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
const_where_expression<simd_mask<T, Abi>, simd<T, Abi>>
where(simd_mask<T, Abi> const& mask, simd<T, Abi> const& value) {
  return const_where_expression(mask, value);
}

template <class T, class Abi>
class scatter {
  simd_index<T, Abi> m_index;
 public:
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  scatter(simd_index<T, Abi> const& index_arg)
   :m_index(index_arg)
  {}
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  simd_index<T, Abi> const& index() const { return m_index; }
};

template <class T, class Abi>
class gather {
  simd_index<T, Abi> m_index;
 public:
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  gather(simd_index<T, Abi> const& index_arg)
   :m_index(index_arg)
  {}
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  simd_index<T, Abi> const& index() const { return m_index; }
};

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

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
simd_index<T, Abi>
operator+(simd_index<T, Abi> const& a, int b) {
  return a + simd_index<T, Abi>(b);
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
simd_index<T, Abi>
operator+(int a, simd_index<T, Abi> const& b) {
  return simd_index<T, Abi>(a) + b;
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
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
simd<T, Abi> load(
    T const* ptr, int offset, simd_mask<T, Abi> const& mask)
{
  simd<T, Abi> result;
  where(mask, result).copy_from(ptr + offset, element_aligned_tag());
  return result;
}

template <class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
simd<T, Abi> load(
    T const* ptr, simd_index<T, Abi> const& offset, simd_mask<T, Abi> const& mask)
{
  simd<T, Abi> result;
  where(mask, result).copy_from(ptr, gather(offset));
  return result;
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
void store(
    simd<T, Abi> const& value,
    T* ptr, int offset, simd_mask<T, Abi> const& mask)
{
  where(mask, value).copy_to(ptr + offset, element_aligned_tag());
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
void store(
    simd<T, Abi> const& value,
    T* ptr, simd_index<T, Abi> const& offset, simd_mask<T, Abi> const& mask)
{
  where(mask, value).copy_to(ptr, scatter(offset));
}

template <class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
T get(simd<T, Abi> const& value, int i)
{
  T storage[simd<T, Abi>::size()];
  value.store(storage);
  return storage[i];
}

// fallback implementations of transcendental functions.
// individual Abi types may provide overloads with more efficient implementations.

template <class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
simd<T, Abi> natural_exponential(simd<T, Abi> const& a)
{
  T a_array[simd<T, Abi>::size()];
  a.store(a_array);
  for (int i = 0; i < simd<T, Abi>::size(); ++i) {
    a_array[i] = natural_exponential(a_array[i]);
  }
  return simd<T, Abi>::load(a_array);
}

template <class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
simd<T, Abi> exponentiate(simd<T, Abi> const& a, simd<T, Abi> const& b)
{
  T a_array[simd<T, Abi>::size()];
  T b_array[simd<T, Abi>::size()];
  a.store(a_array);
  b.store(b_array);
  for (int i = 0; i < simd<T, Abi>::size(); ++i) {
    a_array[i] = exponentiate(a_array[i], b_array[i]);
  }
  return simd<T, Abi>::load(a_array);
}

template <class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
simd<T, Abi> sine(simd<T, Abi> const& a)
{
  T a_array[simd<T, Abi>::size()];
  a.store(a_array);
  for (int i = 0; i < simd<T, Abi>::size(); ++i) {
    a_array[i] = sine(a_array[i]);
  }
  return simd<T, Abi>::load(a_array);
}

template <class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
simd<T, Abi> cosine(simd<T, Abi> const& a)
{
  T a_array[simd<T, Abi>::size()];
  a.store(a_array);
  for (int i = 0; i < simd<T, Abi>::size(); ++i) {
    a_array[i] = cosine(a_array[i]);
  }
  return simd<T, Abi>::load(a_array);
}

}
