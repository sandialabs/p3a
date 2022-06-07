#pragma once

#include <cmath>
#include <cstdint>

#include "p3a_macros.hpp"
#include "p3a_functions.hpp"
#include "p3a_type_traits.hpp"
#include "p3a_scalar.hpp"

namespace p3a {

template <class T, class Abi>
class simd;

template <class T, class Abi>
class simd_mask;

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
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
where_expression<simd_mask<T, Abi>, simd<T, Abi>>
where(no_deduce_t<simd_mask<T, Abi>> const& mask, simd<T, Abi>& value) {
  return where_expression(mask, value);
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
const_where_expression<simd_mask<T, Abi>, simd<T, Abi>>
where(no_deduce_t<simd_mask<T, Abi>> const& mask, simd<T, Abi> const& value) {
  return const_where_expression(mask, value);
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
simd<T, Abi>& operator+=(simd<T, Abi>& a, no_deduce_t<simd<T, Abi>> const& b) {
  a = a + b;
  return a;
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
simd<T, Abi>& operator-=(simd<T, Abi>& a, no_deduce_t<simd<T, Abi>> const& b) {
  a = a - b;
  return a;
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
simd<T, Abi>& operator*=(simd<T, Abi>& a, no_deduce_t<simd<T, Abi>> const& b) {
  a = a * b;
  return a;
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
simd<T, Abi>& operator/=(simd<T, Abi>& a, no_deduce_t<simd<T, Abi>> const& b) {
  a = a / b;
  return a;
}

template <class U, class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
std::enable_if_t<std::is_arithmetic_v<U>, simd<T, Abi>>
operator+(U const& a, simd<T, Abi> const& b) {
  return simd<T, Abi>(T(a)) + b;
}

template <class U, class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
std::enable_if_t<std::is_arithmetic_v<U>, simd<T, Abi>>
operator+(simd<T, Abi> const& a, U const& b) {
  return a + simd<T, Abi>(T(b));
}

template <class U, class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
std::enable_if_t<std::is_arithmetic_v<U>, simd<T, Abi>>
operator-(U const& a, simd<T, Abi> const& b) {
  return simd<T, Abi>(T(a)) - b;
}

template <class U, class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
std::enable_if_t<std::is_arithmetic_v<U>, simd<T, Abi>>
operator-(simd<T, Abi> const& a, U const& b) {
  return a - simd<T, Abi>(T(b));
}

template <class U, class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
std::enable_if_t<std::is_arithmetic_v<U>, simd<T, Abi>>
operator*(U const& a, simd<T, Abi> const& b) {
  return simd<T, Abi>(T(a)) * b;
}

template <class U, class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
std::enable_if_t<std::is_arithmetic_v<U>, simd<T, Abi>>
operator*(simd<T, Abi> const& a, U const& b) {
  return a * simd<T, Abi>(T(b));
}

template <class U, class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
std::enable_if_t<std::is_arithmetic_v<U>, simd<T, Abi>>
operator/(U const& a, simd<T, Abi> const& b) {
  return simd<T, Abi>(T(a)) / b;
}

template <class U, class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
std::enable_if_t<std::is_arithmetic_v<U>, simd<T, Abi>>
operator/(simd<T, Abi> const& a, U const& b) {
  return a / simd<T, Abi>(T(b));
}

template <class U, class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
std::enable_if_t<std::is_arithmetic_v<U>, simd<T, Abi>&>
operator/=(simd<T, Abi>& a, U const& b) {
  a = a / b;
  return a;
}

P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline constexpr bool
all_of(bool a) { return a; }

P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline constexpr bool
any_of(bool a) { return a; }

namespace details {

template <class T, class Abi>
struct is_scalar<simd<T, Abi>> {
  inline static constexpr bool value = true;
};

}

template <class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
simd<T, Abi> load(
    T const* ptr, int offset, simd_mask<T, Abi> const& mask)
{
  simd<T, Abi> result;
  where(mask, result).copy_from(ptr + offset, element_aligned_tag());
  return result;
}

template <class T, class Integral, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
std::enable_if_t<std::is_integral_v<Integral>, simd<T, Abi>>
load(T const* ptr, simd<Integral, Abi> const& offset, simd_mask<T, Abi> const& mask)
{
  simd<T, Abi> result;
  where(mask, result).gather_from(ptr, offset);
  return result;
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
void store(
    simd<T, Abi> const& value,
    T* ptr,
    int offset,
    no_deduce_t<simd_mask<T, Abi>> const& mask)
{
  where(mask, value).copy_to(ptr + offset, element_aligned_tag());
}

template <class T, class Integral, class Abi>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
std::enable_if_t<std::is_integral_v<Integral>, void>
store(
    simd<T, Abi> const& value,
    T* ptr,
    simd<Integral, Abi> const& offset,
    no_deduce_t<simd_mask<T, Abi>> const& mask)
{
  where(mask, value).scatter_to(ptr, offset);
}

template <class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
T get(simd<T, Abi> const& value, int i)
{
  T storage[simd<T, Abi>::size()];
  value.copy_to(storage, element_aligned_tag());
  return storage[i];
}

// fallback implementations of transcendental functions.
// individual Abi types may provide overloads with more efficient implementations.

template <class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
simd<T, Abi> exp(simd<T, Abi> a)
{
  T a_array[simd<T, Abi>::size()];
  a.copy_to(a_array, element_aligned_tag());
  using std::exp;
  for (int i = 0; i < simd<T, Abi>::size(); ++i) {
    a_array[i] = exp(a_array[i]);
  }
  a.copy_from(a_array, element_aligned_tag());
  return a;
}

template <class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
simd<T, Abi> pow(simd<T, Abi> a, simd<T, Abi> const& b)
{
  using std::pow;
  T a_array[simd<T, Abi>::size()];
  T b_array[simd<T, Abi>::size()];
  a.copy_to(a_array, element_aligned_tag());
  b.copy_to(b_array, element_aligned_tag());
  for (int i = 0; i < simd<T, Abi>::size(); ++i) {
    a_array[i] = pow(a_array[i], b_array[i]);
  }
  a.copy_from(a_array, element_aligned_tag());
  return a;
}

template <class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
simd<T, Abi> sin(simd<T, Abi> a)
{
  using std::sin;
  T a_array[simd<T, Abi>::size()];
  a.copy_to(a_array, element_aligned_tag());
  for (int i = 0; i < simd<T, Abi>::size(); ++i) {
    a_array[i] = sin(a_array[i]);
  }
  a.copy_from(a_array, element_aligned_tag());
  return a;
}

template <class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
simd<T, Abi> cos(simd<T, Abi> a)
{
  using std::cos;
  T a_array[simd<T, Abi>::size()];
  a.copy_to(a_array, element_aligned_tag());
  for (int i = 0; i < simd<T, Abi>::size(); ++i) {
    a_array[i] = cos(a_array[i]);
  }
  a.copy_from(a_array, element_aligned_tag());
  return a;
}

}
