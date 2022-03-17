#include "gtest/gtest.h"
#include "p3a_quantity.hpp"

TEST(quantity, multiply) {
  static_assert(std::is_same_v<
      p3a::watt,
      p3a::unit<p3a::dimension<-3, 2, 1>>>,
      "Watt is the SI unit equal to kg * m^2 * s^-3");
  static_assert(std::is_same_v<
      p3a::second,
      p3a::unit<p3a::dimension<1, 0, 0>>>,
      "Second is the SI unit with time dimension");
  static_assert(std::is_same_v<
      p3a::joule,
      p3a::unit<p3a::dimension<-2, 2, 1>>>,
      "Joule is the SI unit equal to kg * m^2 * s^-2");
  static_assert(std::is_same_v<
      p3a::unit_multiply<p3a::watt, p3a::second>,
      p3a::joule>,
      "Watt times second = joule");
  auto a = p3a::watts<double>(1.0) * p3a::seconds<double>(2.0);
  static_assert(std::is_same_v<decltype(a), p3a::joules<double>>,
      "Watts times seconds should be Joules");
  EXPECT_FLOAT_EQ(a.value(), 2.0);
}

TEST(quantity, divide) {
  auto a = p3a::meters<double>(1.0) / p3a::seconds<double>(2.0);
  static_assert(std::is_same_v<decltype(a), p3a::meters_per_second<double>>,
      "meters times seconds should be meters per second");
  EXPECT_FLOAT_EQ(a.value(), 0.5);
}

TEST(quantity, temperature) {
  auto absolute_zero_in_kelvin = p3a::degrees_kelvin<double>(0.0);
  auto absolute_zero_in_celcius =
    p3a::degrees_celcius<double>(absolute_zero_in_kelvin);
  EXPECT_FLOAT_EQ(absolute_zero_in_celcius.value(), -273.15);
  auto absolute_zero_in_fahrenheit =
    p3a::degrees_fahrenheit<double>(absolute_zero_in_kelvin);
  printf("absolute zero in Fahrenheit is %f\n",
      absolute_zero_in_fahrenheit.value());
}
