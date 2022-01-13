#include "gtest/gtest.h"
#include "p3a_fixed_point.hpp"

TEST(fixed_point, one){
  int constexpr count = 10;
  double const values[count] = {
    0.0,
    -0.0,
    1.0,
    420.0,
    -420.0,
    1.0e-20,
    1.0e+20,
    1.0e-320,
    -2.0e+20,
    -3.0e+20
  };
  double nonassociative_sum = 0.0;
  int maximum_exponent = std::numeric_limits<int>::lowest();
  for (int i = 0; i < count; ++i) {
    double const value = values[i];
    int sign_bit;
    int exponent;
    std::uint64_t mantissa;
    p3a::decompose_double(value, sign_bit, exponent, mantissa);
    double const recomposed = p3a::compose_double(sign_bit, exponent, mantissa);
    EXPECT_EQ(value, recomposed);
    std::int64_t significand;
    p3a::decompose_double(value, significand, exponent);
    double const recomposed_again = p3a::compose_double(significand, exponent);
    EXPECT_EQ(value, recomposed_again);
    nonassociative_sum += value;
    maximum_exponent = std::max(maximum_exponent, exponent);
  }
  printf("non-associative sum is %.17e, maximum exponent is %d\n",
      nonassociative_sum, maximum_exponent);
  auto fixed_point_sum_128 = p3a::int128(0);
  for (int i = 0; i < count; ++i) {
    double const value = values[i];
    int exponent;
    std::int64_t significand;
    p3a::decompose_double(value, significand, exponent);
    printf("value %.17e = %lld * (2 ^ %d)\n", value, significand, exponent);
    int const shift = maximum_exponent - exponent;
    printf("needs shift %d to be desired exponent %d\n", shift, maximum_exponent);
    significand = p3a::fixed_point_right_shift(significand, shift);
    printf("value %.17e ~= %lld * (2 ^ %d)\n", value, significand, maximum_exponent);
    fixed_point_sum_128 += p3a::int128(significand);
  }
  printf("fixed point sum high %lld low %llu\n",
      fixed_point_sum_128.high(), fixed_point_sum_128.low());
  int sum_sign;
  if (fixed_point_sum_128 < p3a::int128(0)) {
    sum_sign = -1;
    fixed_point_sum_128 = -fixed_point_sum_128;
  } else {
    sum_sign = 1;
  }
  int sum_exponent = maximum_exponent;
  p3a::int128 const maximum_significand_128(
    0b11111111111111111111111111111111111111111111111111111ll);
  while (fixed_point_sum_128 > maximum_significand_128) {
    printf("happened!\n");
    fixed_point_sum_128 >>= 1;
    ++sum_exponent;
  }
  std::int64_t const fixed_point_sum_64 = sum_sign * p3a::bit_cast<std::int64_t>(fixed_point_sum_128.low());
  printf("fixed point sum = %lld * (2 ^ %d)\n", fixed_point_sum_64, maximum_exponent);
  double const recomposed_fixed_point_sum = p3a::compose_double(fixed_point_sum_64, maximum_exponent);
  printf("recomposed fixed point sum = %.17e\n", recomposed_fixed_point_sum); 
  EXPECT_EQ(recomposed_fixed_point_sum, nonassociative_sum);
}

