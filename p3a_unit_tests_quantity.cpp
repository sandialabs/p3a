#include "gtest/gtest.h"

#include <sstream>

#include "p3a_unit.hpp"
//#include "p3a_quantity.hpp"
//#include "p3a_iostream.hpp"

TEST(quantity, multiply) {
  static_assert(std::is_same_v<
      p3a::watt::dimension,
      p3a::dimension<-3, 2, 1>>,
      "Watt is the SI unit equal to kg * m^2 * s^-3");
  static_assert(std::is_same_v<
      p3a::second::dimension,
      p3a::dimension<1, 0, 0>>,
      "Second is the SI unit with time dimension");
  static_assert(std::is_same_v<
      p3a::joule::dimension,
      p3a::dimension<-2, 2, 1>>,
      "Joule is the SI unit equal to kg * m^2 * s^-2");
  using watt_second = p3a::unit_multiply<p3a::watt, p3a::second>;
  static_assert(p3a::is_same_unit<
      watt_second,
      p3a::joule>,
      "Watt times second = joule");
  EXPECT_EQ(watt_second::name(), "W*s");
  using second_squared = p3a::unit_multiply<p3a::second, p3a::second>;
  EXPECT_EQ(second_squared::name(), "s^2");
  using meter_per_meter = p3a::unit_divide<p3a::meter, p3a::meter>;
  EXPECT_EQ(meter_per_meter::name(), "1");
  using density_times_volume = p3a::unit_multiply<p3a::kilogram_per_cubic_meter, p3a::cubic_meter>;
  EXPECT_EQ(density_times_volume::name(), "kg");
//auto a = p3a::watts<double>(1.0) * p3a::seconds<double>(2.0);
//static_assert(std::is_same_v<decltype(a), p3a::joules<double>>,
//    "Watts times seconds should be Joules");
//EXPECT_FLOAT_EQ(a.value(), 2.0);
}

#if 0

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
  EXPECT_FLOAT_EQ(absolute_zero_in_fahrenheit.value(), -459.67);
  auto const human_fever_temperature_fahrenheit =
    p3a::degrees_fahrenheit<double>(100.4);
  auto const human_fever_temperature_celcius =
    p3a::degrees_celcius<double>(human_fever_temperature_fahrenheit);
  EXPECT_FLOAT_EQ(human_fever_temperature_celcius.value(), 38.0);
  auto const water_freezing_point_celcius =
    p3a::degrees_celcius<double>(0.0);
  auto const water_freezing_point_fahrenheit =
    p3a::degrees_fahrenheit<double>(water_freezing_point_celcius);
  EXPECT_FLOAT_EQ(water_freezing_point_fahrenheit.value(), 32.0);
}

TEST(quantity, percent) {
  auto eighty_percent = p3a::percentage<double>(80.0);
  auto point_eight = p3a::unitless<double>(eighty_percent);
  EXPECT_FLOAT_EQ(point_eight.value(), 0.80);
}

TEST(quantity, thou) {
  using thou = p3a::milli<p3a::inch>;
  auto one_thou = p3a::quantity<thou, double>(1.0);
  auto one_thou_in_micrometers =
    p3a::micrometers<double>(one_thou);
  EXPECT_FLOAT_EQ(one_thou_in_micrometers.value(), 25.4);
}

TEST(quantity, electronvolt) {
  auto const fusion_plasma_temp_in_eV =
    p3a::temperature_electronvolts<double>(15.0e3);
  auto const fusion_plasma_temp_in_K =
    p3a::degrees_kelvin<double>(fusion_plasma_temp_in_eV);
  EXPECT_FLOAT_EQ(
      fusion_plasma_temp_in_K.value(),
      1.74067771800000012e+08);
}

TEST(quantity, cgs) {
  using megagram = p3a::mega<p3a::gram>;
  using megagram_per_cubic_meter =
    p3a::unit_divide<megagram, p3a::cubic_meter>;
  static_assert(std::is_same_v<
      megagram_per_cubic_meter,
      p3a::gram_per_cubic_centimeter>,
      "Mg/m^3 should be the same as g/cm^3");
}

// test that relative and absolute quantities behave
// like points and vectors of an affine space of
// dimension one
//
// https://en.wikipedia.org/wiki/Affine_space

TEST(quantity, affine) {
  using point = p3a::absolute_quantity<p3a::meter>;
  using vector = p3a::quantity<p3a::meter>;
  auto a = point(1.0);
  auto b = point(2.0);
  auto c = point(3.0);
  auto ab = b - a;
  static_assert(std::is_same_v<decltype(ab), vector>,
      "subtracting points should give a vector");
  EXPECT_FLOAT_EQ(ab.value(), 1.0);
  auto b2 = a + ab;
  static_assert(std::is_same_v<decltype(b2), point>,
      "adding a vector to a point should yield a point");
  EXPECT_TRUE(b == b2);
  auto bc = c - b;
  // Weyl's second axiom
  auto ac = ab + bc;
  static_assert(std::is_same_v<decltype(ac), vector>,
      "adding two vectors should yield a vector");
  auto c2 = a + ac;
  EXPECT_TRUE(c == c2);
}

TEST(quantity, gaussian) {
  using gaussian_inverse_seconds =
    p3a::quantity<p3a::gaussian_electrical_conductivity_unit, double>;
  auto a = p3a::siemens_per_meter_quantity<double>(1.0);
  auto b = gaussian_inverse_seconds(a);
  EXPECT_FLOAT_EQ(b.value(), 8.98755178736817551e+09);
}

TEST(quantity, literals) {
  using namespace p3a::quantity_literals;
  auto v = 3.0_m / 1.0_s;
  static_assert(std::is_same_v<p3a::meters_per_second<double>, decltype(v)>,
      "dividing literal meters by literal seconds should yield meters per second");
  EXPECT_FLOAT_EQ(v.value(), 3.0);
}

TEST(quantity, iostream) {
  std::stringstream ss;
  ss << p3a::meters_per_second<double>(3.5);
  auto s = ss.str();
  EXPECT_EQ(s, "3.5 m / s");
}

#endif
