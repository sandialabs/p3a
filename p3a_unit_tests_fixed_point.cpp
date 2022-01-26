#include "gtest/gtest.h"
#include "p3a_fixed_point.hpp"

TEST(fixed_point, sum){
  int constexpr count = 10;
  double const values[count] = {
    0.0,
    -0.0,
    1.0,
    420.0,
    -420.0,
    1.0e-20,
    1.0e+20,
    1.0e-320, //subnormal
    -2.0e+20,
    -3.0e+20
  };
  double nonassociative_sum = 0.0;
  int maximum_exponent = -1075;
  for (int i = 0; i < count; ++i) {
    using abi_type = p3a::simd_abi::host_native;
    p3a::simd<double, abi_type> value;
    auto const mask = p3a::simd_mask<double, abi_type>::first_n(1);
    where(mask, value).copy_from(values + i, p3a::element_aligned_tag());
    p3a::simd<std::int32_t, abi_type> sign_bit;
    p3a::simd<std::int32_t, abi_type> exponent;
    p3a::simd<std::uint64_t, abi_type> mantissa;
    p3a::details::decompose_double(value, sign_bit, exponent, mantissa);
    p3a::simd<double, abi_type> const recomposed =
      p3a::details::compose_double(sign_bit, exponent, mantissa);
    EXPECT_EQ(mask, mask && (recomposed == value));
//  std::int64_t significand;
//  p3a::details::decompose_double(value, significand, exponent);
//  double const recomposed_again = p3a::details::compose_double(significand, exponent);
//  EXPECT_EQ(value, recomposed_again);
//  nonassociative_sum += value;
//  maximum_exponent = std::max(maximum_exponent, exponent);
  }
//auto fixed_point_sum_128 = p3a::details::int128(0);
//for (int i = 0; i < count; ++i) {
//  double const value = values[i];
//  int exponent;
//  std::int64_t significand;
//  p3a::details::decompose_double(value, significand, exponent);
//  significand = p3a::details::decompose_double(value, maximum_exponent);
//  fixed_point_sum_128 += p3a::details::int128(significand);
//}
//double const recomposed_fixed_point_sum = p3a::details::compose_double(fixed_point_sum_128, maximum_exponent);
//// in this small example, the sums are exactly the same
//EXPECT_EQ(recomposed_fixed_point_sum, nonassociative_sum);
}

