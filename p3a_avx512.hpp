#pragma once

#include <immintrin.h>

namespace p3a {

namespace simd_abi {

class avx512 {};

}

template <>
class simd_mask<float, simd_abi::avx512> {
  __mmask16 m_value;
 public:
  using value_type = bool;
  using simd_type = simd<float, simd_abi::avx512>;
  using abi_type = simd_abi::avx512;
  P3A_ALWAYS_INLINE inline simd_mask() = default;
  P3A_ALWAYS_INLINE inline simd_mask(bool value)
    :m_value(-std::int16_t(value))
  {}
  P3A_ALWAYS_INLINE inline static constexpr int size() { return 16; }
  P3A_ALWAYS_INLINE inline constexpr simd_mask(__mmask16 const& value_in)
    :m_value(value_in)
  {}
  P3A_ALWAYS_INLINE inline constexpr __mmask16 get() const { return m_value; }
  P3A_ALWAYS_INLINE inline simd_mask operator||(simd_mask const& other) const {
    return simd_mask(_kor_mask16(m_value, other.m_value));
  }
  P3A_ALWAYS_INLINE inline simd_mask operator&&(simd_mask const& other) const {
    return simd_mask(_kand_mask16(m_value, other.m_value));
  }
  P3A_ALWAYS_INLINE inline simd_mask operator!() const {
    return simd_mask(_knot_mask16(m_value));
  }
  P3A_ALWAYS_INLINE static inline
  simd_mask first_n(int n)
  {
    return simd_mask(__mmask16(std::int16_t((1 << n) - 1)));
  }
};

P3A_ALWAYS_INLINE inline bool all_of(simd_mask<float, simd_abi::avx512> const& a) {
  static const __mmask16 false_value(-std::int16_t(false));
  return _kortestc_mask16_u8(a.get(), false_value);
}

P3A_ALWAYS_INLINE inline bool any_of(simd_mask<float, simd_abi::avx512> const& a) {
  static const __mmask16 false_value(-std::int16_t(false));
  return !_kortestc_mask16_u8(~a.get(), false_value);
}

template <>
class simd<float, simd_abi::avx512> {
  __m512 m_value;
 public:
  P3A_ALWAYS_INLINE simd() = default;
  using value_type = float;
  using abi_type = simd_abi::avx512;
  using mask_type = simd_mask<float, abi_type>;
  P3A_ALWAYS_INLINE inline static constexpr int size() { return 16; }
  P3A_ALWAYS_INLINE inline simd(float value)
    :m_value(_mm512_set1_ps(value))
  {}
  P3A_ALWAYS_INLINE inline simd(
      float a, float b, float c, float d,
      float e, float f, float g, float h,
      float i, float j, float k, float l,
      float m, float n, float o, float p)
    :m_value(_mm512_setr_ps(
          a, b, c, d, e, f, g, h,
          i, j, k, l, m, n, o, p))
  {}
  P3A_ALWAYS_INLINE inline constexpr simd(__m512 const& value_in)
    :m_value(value_in)
  {}
  P3A_ALWAYS_INLINE inline simd operator*(simd const& other) const {
    return simd(_mm512_mul_ps(m_value, other.m_value));
  }
  P3A_ALWAYS_INLINE inline simd operator/(simd const& other) const {
    return simd(_mm512_div_ps(m_value, other.m_value));
  }
  P3A_ALWAYS_INLINE inline simd operator+(simd const& other) const {
    return simd(_mm512_add_ps(m_value, other.m_value));
  }
  P3A_ALWAYS_INLINE inline simd operator-(simd const& other) const {
    return simd(_mm512_sub_ps(m_value, other.m_value));
  }
  P3A_ALWAYS_INLINE inline simd operator-() const {
    return simd(_mm512_sub_ps(_mm512_set1_ps(0.0), m_value));
  }
  P3A_ALWAYS_INLINE static inline simd load(float const* ptr) {
    return simd(_mm512_loadu_ps(ptr));
  }
  P3A_ALWAYS_INLINE inline void store(float* ptr) const {
    _mm512_storeu_ps(ptr, m_value);
  }
  P3A_ALWAYS_INLINE static inline simd masked_load(float const* ptr, mask_type const& mask) {
    return simd(_mm512_mask_loadu_ps(_mm512_set1_ps(0.0), mask.get(), ptr));
  }
  P3A_ALWAYS_INLINE inline void masked_store(float* ptr, mask_type const& mask) const {
    _mm512_mask_storeu_ps(ptr, mask.get(), m_value);
  }
  P3A_ALWAYS_INLINE inline constexpr __m512 get() const { return m_value; }
  P3A_ALWAYS_INLINE inline simd_mask<float, simd_abi::avx512> operator<(simd const& other) const {
    return simd_mask<float, simd_abi::avx512>(_mm512_cmp_ps_mask(m_value, other.m_value, _CMP_LT_OQ));
  }
  P3A_ALWAYS_INLINE inline simd_mask<float, simd_abi::avx512> operator>(simd const& other) const {
    return simd_mask<float, simd_abi::avx512>(_mm512_cmp_ps_mask(m_value, other.m_value, _CMP_GT_OQ));
  }
  P3A_ALWAYS_INLINE inline simd_mask<float, simd_abi::avx512> operator==(simd const& other) const {
    return simd_mask<float, simd_abi::avx512>(_mm512_cmp_ps_mask(m_value, other.m_value, _CMP_EQ_OQ));
  }
  P3A_ALWAYS_INLINE inline simd_mask<float, simd_abi::avx512> operator!=(simd const& other) const {
    return simd_mask<float, simd_abi::avx512>(_mm512_cmp_ps_mask(m_value, other.m_value, _CMP_NEQ_OQ));
  }
  P3A_ALWAYS_INLINE static inline simd zero() {
    return simd(float(0));
  }
};

P3A_ALWAYS_INLINE inline simd<float, simd_abi::avx512> copysign(simd<float, simd_abi::avx512> const& a, simd<float, simd_abi::avx512> const& b) {
  static const __m512i sign_mask = reinterpret_cast<__m512i>(simd<float, simd_abi::avx512>(-0.0).get());
  return simd<float, simd_abi::avx512>(
      reinterpret_cast<__m512>(_mm512_xor_epi32(
          _mm512_andnot_epi32(sign_mask, reinterpret_cast<__m512i>(a.get())),
          _mm512_and_epi32(sign_mask, reinterpret_cast<__m512i>(b.get()))
          ))
      );
}

P3A_ALWAYS_INLINE inline simd<float, simd_abi::avx512> absolute_value(simd<float, simd_abi::avx512> const& a) {
  __m512 const rhs = a.get();
  return reinterpret_cast<__m512>(_mm512_and_epi32(reinterpret_cast<__m512i>(rhs), _mm512_set1_epi32(0x7fffffff)));
}

P3A_ALWAYS_INLINE inline simd<float, simd_abi::avx512> square_root(simd<float, simd_abi::avx512> const& a) {
  return simd<float, simd_abi::avx512>(_mm512_sqrt_ps(a.get()));
}

#ifdef __INTEL_COMPILER
P3A_ALWAYS_INLINE inline simd<float, simd_abi::avx512> cbrt(simd<float, simd_abi::avx512> const& a) {
  return simd<float, simd_abi::avx512>(_mm512_cbrt_ps(a.get()));
}

P3A_ALWAYS_INLINE inline simd<float, simd_abi::avx512> exp(simd<float, simd_abi::avx512> const& a) {
  return simd<float, simd_abi::avx512>(_mm512_exp_ps(a.get()));
}

P3A_ALWAYS_INLINE inline simd<float, simd_abi::avx512> log(simd<float, simd_abi::avx512> const& a) {
  return simd<float, simd_abi::avx512>(_mm512_log_ps(a.get()));
}
#endif

P3A_ALWAYS_INLINE inline simd<float, simd_abi::avx512> fma(
    simd<float, simd_abi::avx512> const& a,
    simd<float, simd_abi::avx512> const& b,
    simd<float, simd_abi::avx512> const& c) {
  return simd<float, simd_abi::avx512>(_mm512_fmadd_ps(a.get(), b.get(), c.get()));
}

P3A_ALWAYS_INLINE inline
simd<float, simd_abi::avx512>
maximum(
    simd<float, simd_abi::avx512> const& a,
    simd<float, simd_abi::avx512> const& b)
{
  return simd<float, simd_abi::avx512>(_mm512_max_ps(a.get(), b.get()));
}

P3A_ALWAYS_INLINE inline
simd<float, simd_abi::avx512>
minimum(
    simd<float, simd_abi::avx512> const& a,
    simd<float, simd_abi::avx512> const& b)
{
  return simd<float, simd_abi::avx512>(_mm512_min_ps(a.get(), b.get()));
}

P3A_ALWAYS_INLINE inline simd<float, simd_abi::avx512>
condition(
    simd_mask<float, simd_abi::avx512> const& a,
    simd<float, simd_abi::avx512> const& b,
    simd<float, simd_abi::avx512> const& c)
{
  return simd<float, simd_abi::avx512>(_mm512_mask_blend_ps(a.get(), c.get(), b.get()));
}

template <>
class simd_mask<double, simd_abi::avx512> {
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
    return simd_mask(static_cast<__mmask8>(_mm512_kor(m_value, other.m_value)));
  }
  P3A_ALWAYS_INLINE simd_mask operator&&(simd_mask const& other) const {
    return simd_mask(static_cast<__mmask8>(_mm512_kand(m_value, other.m_value)));
  }
  P3A_ALWAYS_INLINE simd_mask operator!() const {
    static const __mmask8 true_value(simd_mask<double, simd_abi::avx512>(true).get());
    return simd_mask(static_cast<__mmask8>(_mm512_kxor(true_value, m_value)));
  }
  P3A_ALWAYS_INLINE static inline
  simd_mask first_n(int n)
  {
    return simd_mask(__mmask8(std::int16_t((1 << n) - 1)));
  }
};

P3A_ALWAYS_INLINE inline bool all_of(simd_mask<double, simd_abi::avx512> const& a) {
  static const __mmask16 false_value(-std::int16_t(false));
  const __mmask16 a_value(0xFF00 | a.get());
  return _kortestc_mask16_u8(a_value, false_value);
}

P3A_ALWAYS_INLINE inline bool any_of(simd_mask<double, simd_abi::avx512> const& a) {
  static const __mmask16 false_value(-std::int16_t(false));
  const __mmask16 a_value(0x0000 | a.get());
  return !_kortestc_mask16_u8(~a_value, false_value);
}

P3A_ALWAYS_INLINE inline bool none_of(simd_mask<double, simd_abi::avx512> const& a) {
  return a.get() == simd_mask<double, simd_abi::avx512>(false).get();
}

template <>
class simd_index<double, simd_abi::avx512> {
  __m256i m_value;
 public:
  using value_type = int;
  using abi_type = simd_abi::avx512;
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
    return _mm256_maskz_mul_epi32(mask_type(true).get(), m_value, other.m_value);
  }
  P3A_ALWAYS_INLINE inline simd_index operator+(simd_index const& other) const {
    return _mm256_maskz_add_epi32(mask_type(true).get(), m_value, other.m_value);
  }
  P3A_ALWAYS_INLINE inline simd_index operator-(simd_index const& other) const {
    return _mm256_maskz_sub_epi32(mask_type(true).get(), m_value, other.m_value);
  }
  P3A_ALWAYS_INLINE inline simd_index operator-() const {
    return simd_index(0) - *this;
  }
  P3A_ALWAYS_INLINE inline constexpr __m256i get() const { return m_value; }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512> operator<(simd_index const& other) const {
    return simd_mask<double, simd_abi::avx512>(_mm256_cmp_epi32_mask(m_value, other.m_value, _MM_CMPINT_LT));
  }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512> operator>(simd_index const& other) const {
    return simd_mask<double, simd_abi::avx512>(_mm256_cmp_epi32_mask(other.m_value, m_value, _MM_CMPINT_LT));
  }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512> operator<=(simd_index const& other) const {
    return simd_mask<double, simd_abi::avx512>(_mm256_cmp_epi32_mask(m_value, other.m_value, _MM_CMPINT_LE));
  }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512> operator>=(simd_index const& other) const {
    return simd_mask<double, simd_abi::avx512>(_mm256_cmp_epi32_mask(other.m_value, m_value, _MM_CMPINT_LE));
  }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512> operator==(simd_index const& other) const {
    return simd_mask<double, simd_abi::avx512>(_mm256_cmp_epi32_mask(m_value, other.m_value, _MM_CMPINT_EQ));
  }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512> operator!=(simd_index const& other) const {
    return simd_mask<double, simd_abi::avx512>(_mm256_cmp_epi32_mask(m_value, other.m_value, _MM_CMPINT_NE));
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

template <>
class simd<double, simd_abi::avx512> {
  __m512d m_value;
 public:
  using value_type = double;
  using abi_type = simd_abi::avx512;
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
  P3A_ALWAYS_INLINE static inline simd load(double const* ptr) {
    return simd(_mm512_loadu_pd(ptr));
  }
  P3A_ALWAYS_INLINE inline void store(double* ptr) const {
    _mm512_storeu_pd(ptr, m_value);
  }
  P3A_ALWAYS_INLINE static inline simd masked_load(double const* ptr, mask_type const& mask) {
    return simd(_mm512_mask_loadu_pd(_mm512_set1_pd(0.0), mask.get(), ptr));
  }
  P3A_ALWAYS_INLINE inline void masked_store(double* ptr, mask_type const& mask) const {
    _mm512_mask_storeu_pd(ptr, mask.get(), m_value);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE static inline
  simd masked_gather(double const* ptr, mask_type const& mask, index_type const& index) {
    return _mm512_mask_i32gather_pd(
        _mm512_set1_pd(0.0),
        mask.get(),
        index.get(),
        ptr,
        8);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  void masked_scatter(double* ptr, mask_type const& mask, index_type const& index) const {
    _mm512_mask_i32scatter_pd(
        ptr,
        mask.get(),
        index.get(),
        m_value,
        8);
  }
  P3A_ALWAYS_INLINE inline constexpr __m512d get() const { return m_value; }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512> operator<(simd const& other) const {
    return simd_mask<double, simd_abi::avx512>(_mm512_cmp_pd_mask(m_value, other.m_value, _CMP_LT_OQ));
  }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512> operator>(simd const& other) const {
    return simd_mask<double, simd_abi::avx512>(_mm512_cmp_pd_mask(m_value, other.m_value, _CMP_GT_OQ));
  }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512> operator==(simd const& other) const {
    return simd_mask<double, simd_abi::avx512>(_mm512_cmp_pd_mask(m_value, other.m_value, _CMP_EQ_OQ));
  }
  P3A_ALWAYS_INLINE inline simd_mask<double, simd_abi::avx512> operator!=(simd const& other) const {
    return simd_mask<double, simd_abi::avx512>(_mm512_cmp_pd_mask(m_value, other.m_value, _CMP_NEQ_OQ));
  }
  P3A_ALWAYS_INLINE static inline simd zero() {
    return simd(double(0));
  }
};

P3A_ALWAYS_INLINE inline simd<double, simd_abi::avx512> copysign(simd<double, simd_abi::avx512> const& a, simd<double, simd_abi::avx512> const& b) {
  static const __m512i sign_mask = reinterpret_cast<__m512i>(simd<double, simd_abi::avx512>(-0.0).get());
  return simd<double, simd_abi::avx512>(
      reinterpret_cast<__m512d>(_mm512_xor_epi64(
          _mm512_andnot_epi64(sign_mask, reinterpret_cast<__m512i>(a.get())),
          _mm512_and_epi64(sign_mask, reinterpret_cast<__m512i>(b.get()))
          ))
      );
}

P3A_ALWAYS_INLINE inline simd<double, simd_abi::avx512> absolute_value(simd<double, simd_abi::avx512> const& a) {
  __m512d const rhs = a.get();
  return reinterpret_cast<__m512d>(_mm512_and_epi64(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFF),
        reinterpret_cast<__m512i>(rhs)));
}

P3A_ALWAYS_INLINE inline simd<double, simd_abi::avx512> square_root(simd<double, simd_abi::avx512> const& a) {
  return simd<double, simd_abi::avx512>(_mm512_sqrt_pd(a.get()));
}

#ifdef __INTEL_COMPILER
P3A_ALWAYS_INLINE inline simd<double, simd_abi::avx512> cbrt(simd<double, simd_abi::avx512> const& a) {
  return simd<double, simd_abi::avx512>(_mm512_cbrt_pd(a.get()));
}

P3A_ALWAYS_INLINE inline simd<double, simd_abi::avx512> exp(simd<double, simd_abi::avx512> const& a) {
  return simd<double, simd_abi::avx512>(_mm512_exp_pd(a.get()));
}

P3A_ALWAYS_INLINE inline simd<double, simd_abi::avx512> log(simd<double, simd_abi::avx512> const& a) {
  return simd<double, simd_abi::avx512>(_mm512_log_pd(a.get()));
}
#endif

P3A_ALWAYS_INLINE inline simd<double, simd_abi::avx512> fma(
    simd<double, simd_abi::avx512> const& a,
    simd<double, simd_abi::avx512> const& b,
    simd<double, simd_abi::avx512> const& c) {
  return simd<double, simd_abi::avx512>(_mm512_fmadd_pd(a.get(), b.get(), c.get()));
}

P3A_ALWAYS_INLINE inline
simd<double, simd_abi::avx512>
maximum(
    simd<double, simd_abi::avx512> const& a,
    simd<double, simd_abi::avx512> const& b)
{
  return simd<double, simd_abi::avx512>(_mm512_max_pd(a.get(), b.get()));
}

P3A_ALWAYS_INLINE inline
simd<double, simd_abi::avx512>
minimum(
    simd<double, simd_abi::avx512> const& a,
    simd<double, simd_abi::avx512> const& b)
{
  return simd<double, simd_abi::avx512>(_mm512_min_pd(a.get(), b.get()));
}

P3A_ALWAYS_INLINE inline
simd<double, simd_abi::avx512>
condition(
    simd_mask<double, simd_abi::avx512> const& a,
    simd<double, simd_abi::avx512> const& b,
    simd<double, simd_abi::avx512> const& c)
{
  return simd<double, simd_abi::avx512>(_mm512_mask_blend_pd(a.get(), c.get(), b.get()));
}

}
