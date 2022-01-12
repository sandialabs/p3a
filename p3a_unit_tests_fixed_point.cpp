#include "gtest/gtest.h"
#include "p3a_fixed_point.hpp"

TEST(fixed_point, one){
  int constexpr count = 10;
  double const values[count] = {
    0.0,
    -0.0,
    1.0,
    42.0,
    -42.0,
    1.0e-20,
    -1.0e+20,
    1.0e-320,
    2.0e+20,
    3.0e+20
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
  std::int64_t fixed_point_sum_64 = 0;
  for (int i = 0; i < count; ++i) {
    double const value = values[i];
    int exponent;
    std::int64_t significand;
    p3a::decompose_double(value, significand, exponent);
    printf("value %.17e = %lld * (2 ^ %d)\n", value, significand, exponent);
    int const shift = maximum_exponent - exponent;
    printf("needs shift %d to be desired exponent %d\n", shift, maximum_exponent);
    if (shift > 64) {
      significand = 0;
    } else {
      significand >>= shift;
    }
    printf("value %.17e ~= %lld * (2 ^ %d)\n", value, significand, maximum_exponent);
    fixed_point_sum_64 += significand;
  }
  printf("fixed point sum = %lld * (2 ^ %d)\n", fixed_point_sum_64, maximum_exponent);
}

