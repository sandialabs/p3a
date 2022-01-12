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
  for (int i = 0; i < count; ++i) {
    double const value = values[i];
    int sign_bit;
    int exponent;
    std::uint64_t mantissa;
    p3a::decompose_double(value, sign_bit, exponent, mantissa);
    printf("value %.17e has sign bit %d exponent %d mantissa %llu\n",
        value, sign_bit, exponent, mantissa);
    double const recomposed = p3a::compose_double(sign_bit, exponent, mantissa);
    printf("recomposed value is %.17e\n", recomposed);
    EXPECT_EQ(value, recomposed);
    std::int64_t significand;
    p3a::decompose_double(value, significand, exponent);
    printf("value %.17e = %lld * 2 ^ %d\n", value, significand, exponent);
  }
}

