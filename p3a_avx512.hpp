#pragma once

#include <immintrin.h>

#ifndef __AVX512DQ__
#error "P3A requires AVX512DQ"
#endif

#include "p3a_functional.hpp"

namespace p3a {

namespace simd_abi {

class avx512_mm512 {};

}

template <class T>
class simd_mask<T, std::enable_if_t<sizeof(T) == 8, simd_abi::avx512_mm512>> {
  __mmask8 m_value;
 public:
  using value_type = bool;
  P3A_ALWAYS_INLINE inline simd_mask() = default;
  P3A_ALWAYS_INLINE inline simd_mask(bool value)
    :m_value(-std::int16_t(value))
  {}
  P3A_ALWAYS_INLINE inline static constexpr int size() { return 8; }
  P3A_ALWAYS_INLINE inline constexpr simd_mask(__mmask8 const& value_in)
    :m_value(value_in)
  {}
  P3A_ALWAYS_INLINE inline constexpr __mmask8 get() const { return m_value; }
  P3A_ALWAYS_INLINE simd_mask operator||(simd_mask const& other) const {
    return simd_mask(_kor_mask8(m_value, other.m_value));
  }
  P3A_ALWAYS_INLINE simd_mask operator&&(simd_mask const& other) const {
    return simd_mask(_kand_mask8(m_value, other.m_value));
  }
  P3A_ALWAYS_INLINE simd_mask operator!() const {
    static const __mmask8 true_value(simd_mask<T, simd_abi::avx512_mm512>(true).get());
    return simd_mask(_kxor_mask8(true_value, m_value));
  }
  P3A_ALWAYS_INLINE static inline
  simd_mask first_n(int n)
  {
    return simd_mask(__mmask8(std::int16_t((1 << n) - 1)));
  }
};

template <class T>
P3A_ALWAYS_INLINE inline
std::enable_if_t<sizeof(T) == 8, bool>
all_of(simd_mask<T, simd_abi::avx512_mm512> const& a) {
  static const __mmask16 false_value(-std::int16_t(false));
  const __mmask16 a_value(0xFF00 | a.get());
  return _kortestc_mask16_u8(a_value, false_value);
}

template <class T>
P3A_ALWAYS_INLINE inline
std::enable_if_t<sizeof(T) == 8, bool>
any_of(simd_mask<T, simd_abi::avx512_mm512> const& a) {
  static const __mmask16 false_value(-std::int16_t(false));
  const __mmask16 a_value(0x0000 | a.get());
  return !_kortestc_mask16_u8(~a_value, false_value);
}

template <class T>
P3A_ALWAYS_INLINE inline
std::enable_if_t<sizeof(T) == 8, bool>
none_of(simd_mask<T, simd_abi::avx512_mm512> const& a) {
  return a.get() == simd_mask<T, simd_abi::avx512_mm512>(false).get();
}

template <>
class simd_index<double, simd_abi::avx512_mm512> {
  __m256i m_value;
 public:
  using value_type = int;
  using abi_type = simd_abi::avx512_mm512;
  using mask_type = simd_mask<double, abi_type>;
  P3A_ALWAYS_INLINE inline simd_index() = default;
  P3A_ALWAYS_INLINE inline simd_index(simd_index const&) = default;
  P3A_ALWAYS_INLINE inline simd_index(simd_index&&) = default;
  P3A_ALWAYS_INLINE inline simd_index& operator=(simd_index const&) = default;
  P3A_ALWAYS_INLINE inline simd_index& operator=(simd_index&&) = default;
  P3A_ALWAYS_INLINE inline static constexpr int size() { return 8; }
  P3A_ALWAYS_INLINE inline simd_index(int value)
    :m_value(_mm256_set1_epi32(value))
  {}
  P3A_ALWAYS_INLINE inline constexpr simd_index(__m256i const& value_in)
    :m_value(value_in)
  {}
  P3A_ALWAYS_INLINE inline simd_index operator*(simd_index const& other) const {
    return _mm256_mullo_epi32(m_value, other.m_value);
  }
  P3A_ALWAYS_INLINE inline simd_index operator+(simd_index const& other) const {
    return _mm256_add_epi32(m_value, other.m_value);
  }
  P3A_ALWAYS_INLINE inline simd_index operator-(simd_index const& other) const {
    return _mm256_sub_epi32(m_value, other.m_value);
  }
  P3A_ALWAYS_INLINE inline simd_index operator-() const {
    return simd_index(0) - *this;
  }
  P3A_ALWAYS_INLINE inline constexpr __m256i get() const { return m_value; }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512_mm512> operator<(simd_index const& other) const {
    return simd_mask<double, simd_abi::avx512_mm512>(_mm256_cmplt_epi32_mask(m_value, other.m_value));
  }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512_mm512> operator>(simd_index const& other) const {
    return simd_mask<double, simd_abi::avx512_mm512>(_mm256_cmplt_epi32_mask(other.m_value, m_value));
  }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512_mm512> operator<=(simd_index const& other) const {
    return simd_mask<double, simd_abi::avx512_mm512>(_mm256_cmple_epi32_mask(m_value, other.m_value));
  }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512_mm512> operator>=(simd_index const& other) const {
    return simd_mask<double, simd_abi::avx512_mm512>(_mm256_cmple_epi32_mask(other.m_value, m_value));
  }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512_mm512> operator==(simd_index const& other) const {
    return simd_mask<double, simd_abi::avx512_mm512>(_mm256_cmpeq_epi32_mask(m_value, other.m_value));
  }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512_mm512> operator!=(simd_index const& other) const {
    return simd_mask<double, simd_abi::avx512_mm512>(_mm256_cmpneq_epi32_mask(m_value, other.m_value));
  }
  P3A_ALWAYS_INLINE inline void masked_store(int* ptr, mask_type const& mask) const {
    _mm256_mask_storeu_epi32(ptr, mask.get(), m_value);
  }
  P3A_ALWAYS_INLINE static inline simd_index contiguous_from(int i) {
    return _mm256_setr_epi32(
        i,
        i + 1,
        i + 2,
        i + 3,
        i + 4,
        i + 5,
        i + 6,
        i + 7);
  }
};

P3A_ALWAYS_INLINE inline
simd_index<double, simd_abi::avx512_mm512>
condition(
    simd_mask<double, simd_abi::avx512_mm512> const& a,
    simd_index<double, simd_abi::avx512_mm512> const& b,
    simd_index<double, simd_abi::avx512_mm512> const& c)
{
  return simd_index<double, simd_abi::avx512_mm512>(_mm256_mask_blend_epi32(a.get(), c.get(), b.get()));
}

template <>
class simd<double, simd_abi::avx512_mm512> {
  __m512d m_value;
 public:
  using value_type = double;
  using abi_type = simd_abi::avx512_mm512;
  using mask_type = simd_mask<double, abi_type>;
  using index_type = simd_index<double, abi_type>;
  P3A_ALWAYS_INLINE inline simd() = default;
  P3A_ALWAYS_INLINE inline simd(simd const&) = default;
  P3A_ALWAYS_INLINE inline simd(simd&&) = default;
  P3A_ALWAYS_INLINE inline simd& operator=(simd const&) = default;
  P3A_ALWAYS_INLINE inline simd& operator=(simd&&) = default;
  P3A_ALWAYS_INLINE inline static constexpr int size() { return 8; }
  P3A_ALWAYS_INLINE inline simd(double value)
    :m_value(_mm512_set1_pd(value))
  {}
  P3A_ALWAYS_INLINE inline simd(
      double a, double b, double c, double d,
      double e, double f, double g, double h)
    :m_value(_mm512_setr_pd(a, b, c, d, e, f, g, h))
  {}
  P3A_ALWAYS_INLINE inline constexpr simd(__m512d const& value_in)
    :m_value(value_in)
  {}
  P3A_ALWAYS_INLINE inline simd operator*(simd const& other) const {
    return simd(_mm512_mul_pd(m_value, other.m_value));
  }
  P3A_ALWAYS_INLINE inline simd operator/(simd const& other) const {
    return simd(_mm512_div_pd(m_value, other.m_value));
  }
  P3A_ALWAYS_INLINE inline simd operator+(simd const& other) const {
    return simd(_mm512_add_pd(m_value, other.m_value));
  }
  P3A_ALWAYS_INLINE inline simd operator-(simd const& other) const {
    return simd(_mm512_sub_pd(m_value, other.m_value));
  }
  P3A_ALWAYS_INLINE inline simd operator-() const {
    return simd(_mm512_sub_pd(_mm512_set1_pd(0.0), m_value));
  }
  P3A_ALWAYS_INLINE inline void copy_from(double const* ptr, element_aligned_tag) {
    m_value = _mm512_loadu_pd(ptr);
  }
  P3A_ALWAYS_INLINE inline void copy_to(double* ptr, element_aligned_tag) const {
    _mm512_storeu_pd(ptr, m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE static inline
  simd masked_gather(double const* ptr, index_type const& index, mask_type const& mask) {
    return _mm512_mask_i32gather_pd(
        _mm512_set1_pd(0.0),
        mask.get(),
        index.get(),
        ptr,
        8);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  void masked_scatter(double* ptr, index_type const& index, mask_type const& mask) const {
    _mm512_mask_i32scatter_pd(
        ptr,
        mask.get(),
        index.get(),
        m_value,
        8);
  }
  P3A_ALWAYS_INLINE inline constexpr __m512d get() const { return m_value; }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512_mm512> operator<(simd const& other) const {
    return simd_mask<double, simd_abi::avx512_mm512>(_mm512_cmplt_pd_mask(m_value, other.m_value));
  }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512_mm512> operator>(simd const& other) const {
    return simd_mask<double, simd_abi::avx512_mm512>(_mm512_cmplt_pd_mask(other.m_value, m_value));
  }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512_mm512> operator<=(simd const& other) const {
    return simd_mask<double, simd_abi::avx512_mm512>(_mm512_cmple_pd_mask(m_value, other.m_value));
  }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512_mm512> operator>=(simd const& other) const {
    return simd_mask<double, simd_abi::avx512_mm512>(_mm512_cmple_pd_mask(other.m_value, m_value));
  }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512_mm512> operator==(simd const& other) const {
    return simd_mask<double, simd_abi::avx512_mm512>(_mm512_cmpeq_pd_mask(m_value, other.m_value));
  }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512_mm512> operator!=(simd const& other) const {
    return simd_mask<double, simd_abi::avx512_mm512>(_mm512_cmpneq_pd_mask(m_value, other.m_value));
  }
  P3A_ALWAYS_INLINE static inline simd zero() {
    return simd(double(0));
  }
};

P3A_ALWAYS_INLINE inline simd<double, simd_abi::avx512_mm512> copysign(simd<double, simd_abi::avx512_mm512> const& a, simd<double, simd_abi::avx512_mm512> const& b) {
  static const __m512i sign_mask = reinterpret_cast<__m512i>(simd<double, simd_abi::avx512_mm512>(-0.0).get());
  return simd<double, simd_abi::avx512_mm512>(
      reinterpret_cast<__m512d>(_mm512_xor_epi64(
          _mm512_andnot_epi64(sign_mask, reinterpret_cast<__m512i>(a.get())),
          _mm512_and_epi64(sign_mask, reinterpret_cast<__m512i>(b.get()))
          ))
      );
}

P3A_ALWAYS_INLINE inline simd<double, simd_abi::avx512_mm512> absolute_value(simd<double, simd_abi::avx512_mm512> const& a) {
  __m512d const rhs = a.get();
  return reinterpret_cast<__m512d>(_mm512_and_epi64(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFF),
        reinterpret_cast<__m512i>(rhs)));
}

P3A_ALWAYS_INLINE inline simd<double, simd_abi::avx512_mm512> square_root(simd<double, simd_abi::avx512_mm512> const& a) {
  return simd<double, simd_abi::avx512_mm512>(_mm512_sqrt_pd(a.get()));
}

#ifdef __INTEL_COMPILER
P3A_ALWAYS_INLINE inline simd<double, simd_abi::avx512_mm512> cbrt(simd<double, simd_abi::avx512_mm512> const& a) {
  return simd<double, simd_abi::avx512_mm512>(_mm512_cbrt_pd(a.get()));
}

P3A_ALWAYS_INLINE inline simd<double, simd_abi::avx512_mm512> exp(simd<double, simd_abi::avx512_mm512> const& a) {
  return simd<double, simd_abi::avx512_mm512>(_mm512_exp_pd(a.get()));
}

P3A_ALWAYS_INLINE inline simd<double, simd_abi::avx512_mm512> log(simd<double, simd_abi::avx512_mm512> const& a) {
  return simd<double, simd_abi::avx512_mm512>(_mm512_log_pd(a.get()));
}
#endif

P3A_ALWAYS_INLINE inline simd<double, simd_abi::avx512_mm512> fma(
    simd<double, simd_abi::avx512_mm512> const& a,
    simd<double, simd_abi::avx512_mm512> const& b,
    simd<double, simd_abi::avx512_mm512> const& c) {
  return simd<double, simd_abi::avx512_mm512>(_mm512_fmadd_pd(a.get(), b.get(), c.get()));
}

P3A_ALWAYS_INLINE inline
simd<double, simd_abi::avx512_mm512>
maximum(
    simd<double, simd_abi::avx512_mm512> const& a,
    simd<double, simd_abi::avx512_mm512> const& b)
{
  return simd<double, simd_abi::avx512_mm512>(_mm512_max_pd(a.get(), b.get()));
}

P3A_ALWAYS_INLINE inline
simd<double, simd_abi::avx512_mm512>
minimum(
    simd<double, simd_abi::avx512_mm512> const& a,
    simd<double, simd_abi::avx512_mm512> const& b)
{
  return simd<double, simd_abi::avx512_mm512>(_mm512_min_pd(a.get(), b.get()));
}

P3A_ALWAYS_INLINE inline
simd<double, simd_abi::avx512_mm512>
condition(
    simd_mask<double, simd_abi::avx512_mm512> const& a,
    simd<double, simd_abi::avx512_mm512> const& b,
    simd<double, simd_abi::avx512_mm512> const& c)
{
  return simd<double, simd_abi::avx512_mm512>(_mm512_mask_blend_pd(a.get(), c.get(), b.get()));
}

template <>
class const_where_expression<simd_mask<double, simd_abi::avx512_mm512>, simd<double, simd_abi::avx512_mm512>> {
 public:
  using abi_type = simd_abi::avx512_mm512;
  using value_type = simd<double, abi_type>;
  using mask_type = simd_mask<double, abi_type>;
 protected:
  value_type& m_value;
  mask_type const& m_mask;
 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
    :m_value(const_cast<value_type&>(value_arg))
    ,m_mask(mask_arg)
  {}
  [[nodiscard]] P3A_ALWAYS_INLINE inline constexpr
  mask_type const& mask() const { return m_mask; }
  [[nodiscard]] P3A_ALWAYS_INLINE inline constexpr
  value_type const& value() const { return m_value; }
  P3A_ALWAYS_INLINE inline
  void copy_to(double* mem, element_aligned_tag) const {
    _mm512_mask_storeu_pd(mem, m_mask.get(), m_value.get());
  }
  P3A_ALWAYS_INLINE inline
  void scatter_to(double* mem, simd_index<double, simd_abi::avx512_mm512> const& index) const {
    _mm512_mask_i32scatter_pd(
        mem,
        m_mask.get(),
        index.get(),
        m_value.get(),
        8);
  }
};

template <>
class where_expression<simd_mask<double, simd_abi::avx512_mm512>, simd<double, simd_abi::avx512_mm512>>
 : public const_where_expression<simd_mask<double, simd_abi::avx512_mm512>, simd<double, simd_abi::avx512_mm512>> {
 public:
  where_expression(simd_mask<double, simd_abi::avx512_mm512> const& mask_arg, simd<double, simd_abi::avx512_mm512>& value_arg)
    :const_where_expression(mask_arg, value_arg)
  {}
  P3A_ALWAYS_INLINE inline
  void copy_from(double const* mem, element_aligned_tag) {
    m_value = value_type(_mm512_mask_loadu_pd(_mm512_set1_pd(0.0), m_mask.get(), mem));
  }
  P3A_ALWAYS_INLINE inline
  void gather_from(double const* mem, simd_index<double, simd_abi::avx512_mm512> const& index) {
    m_value = value_type(_mm512_mask_i32gather_pd(
        _mm512_set1_pd(0.0),
        m_mask.get(),
        index.get(),
        mem,
        8));
  }
};

[[nodiscard]] P3A_ALWAYS_INLINE inline
double reduce(
    const_where_expression<simd_mask<double, simd_abi::avx512_mm512>, simd<double, simd_abi::avx512_mm512>> const& x,
    double,
    minimizer<double>)
{
  return _mm512_mask_reduce_min_pd(x.mask().get(), x.value().get());
}

[[nodiscard]] P3A_ALWAYS_INLINE inline
double reduce(
    const_where_expression<simd_mask<double, simd_abi::avx512_mm512>, simd<double, simd_abi::avx512_mm512>> const& x,
    double,
    adder<double>)
{
  return _mm512_mask_reduce_add_pd(x.mask().get(), x.value().get());
}

}
