#include "kul.hpp"

#include "gtest/gtest.h"

#include <iostream>

TEST(rational, construct)
{
  auto a = kul::rational(1);
  EXPECT_EQ(a.numerator(), 1);
  EXPECT_EQ(a.denominator(), 1);
  auto b = kul::rational(5);
  EXPECT_EQ(b.numerator(), 5);
  EXPECT_EQ(b.denominator(), 1);
}

TEST(rational, simplify)
{
  auto a = kul::rational(10, 30);
  EXPECT_EQ(a.numerator(), 1);
  EXPECT_EQ(a.denominator(), 3);
}

TEST(rational, compare)
{
  auto a = kul::rational(5, 3);
  auto b = kul::rational(5, 3);
  auto c = kul::rational(3, 5);
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

TEST(dimension, compare)
{
  auto a = kul::time();
  auto b = kul::time();
  auto c = kul::length();
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

TEST(named_unit, second)
{
  static_assert(kul::second::static_dimension() == kul::time(), "second has dimension time");
  static_assert(kul::second::static_magnitude() == kul::rational(1), "second is an SI unit");
  static_assert(!kul::second::static_origin().has_value(), "second is relative");
  kul::second s;
  EXPECT_EQ(s.name(), "s");
  EXPECT_EQ(s.dimension(), kul::time());
  EXPECT_EQ(s.magnitude(), kul::rational(1));
  EXPECT_FALSE(s.origin().has_value());
}

TEST(named_unit, compare)
{
  kul::second a;
  kul::second b;
  EXPECT_EQ(a, b);
  kul::meter c;
  EXPECT_NE(a.dimension(), c.dimension());
  EXPECT_NE(a, c);
}

TEST(dynamic_product, push_back)
{
  kul::dynamic_product p;
  p.push_back(kul::meter());
  p.push_back(kul::meter());
  p.push_back(kul::unitless());
  EXPECT_EQ(p.name(), "m * m * 1");
}

TEST(dynamic_product, push_back_unless_unitless)
{
  kul::dynamic_product p;
  p.push_back_unless_unitless(kul::meter());
  p.push_back_unless_unitless(kul::meter());
  p.push_back_unless_unitless(kul::unitless());
  EXPECT_EQ(p.name(), "m * m");
}

TEST(dynamic_product, multiply_with_named)
{
  kul::dynamic_product p;
  p.multiply_with(kul::meter());
  EXPECT_EQ(p.name(), "m^1");
}

TEST(dynamic_product, multiply_same_base)
{
  kul::dynamic_product p;
  p.multiply_with(kul::meter());
  p.multiply_with(kul::meter());
  EXPECT_EQ(p.name(), "m^2");
}

TEST(dynamic_product, multiply_alternating_bases)
{
  kul::dynamic_product p;
  p.multiply_with(kul::meter());
  p.multiply_with(kul::second());
  p.multiply_with(kul::meter());
  p.multiply_with(kul::second());
  EXPECT_EQ(p.name(), "m^2 * s^2");
}

TEST(dynamic_product, simplify_exponent_one)
{
  kul::dynamic_product p;
  p.multiply_with(kul::meter());
  EXPECT_EQ(p.name(), "m^1");
  auto s = p.simplify();
  EXPECT_EQ(s->name(), "m");
}

TEST(dynamic_product, simplify_exponent_zero)
{
  kul::dynamic_product p;
  p.multiply_with(kul::meter());
  p.multiply_with(kul::dynamic_exp(kul::meter(), -1));
  EXPECT_EQ(p.name(), "m^0");
  auto s = p.simplify();
  EXPECT_EQ(s->name(), "1");
}

TEST(dynamic_product, multiply_by_simplified)
{
  kul::dynamic_product a;
  a.multiply_with(kul::meter());
  a.multiply_with(kul::second());
  a.multiply_with(kul::second());
  auto b = a.simplify();
  EXPECT_EQ(b->name(), "m * s^2");
  kul::dynamic_product c;
  c.multiply_with(kul::meter());
  auto d = c.simplify();
  EXPECT_EQ(d->name(), "m");
  kul::dynamic_product e;
  e.multiply_with(b);
  e.multiply_with(d);
  EXPECT_EQ(e.name(), "m^2 * s^2");
}

TEST(dynamic_product, m_per_s2)
{
  kul::dynamic_product p;
  p.multiply_with(kul::meter());
  p.divide_by(kul::second());
  p.divide_by(kul::second());
  auto s = p.simplify();
  EXPECT_EQ(s->name(), "m * s^-2");
}

TEST(multiply_divide, m_per_s2)
{
  auto m_per_s2 = kul::meter() / (kul::second() * kul::second());
  EXPECT_EQ(m_per_s2.name(), "m * s^-2");
}

TEST(kilo, kg)
{
  static_assert(kul::kilo<kul::gram>::static_dimension() == kul::mass(), "kilogram is mass");
  static_assert(kul::kilo<kul::gram>::static_magnitude() == kul::rational(1), "kilogram is SI base");
  EXPECT_EQ(kul::kilo<kul::gram>::static_name(), "kg");
}

TEST(static_product, m_s)
{
  using prod = kul::static_product<kul::meter, kul::second>;
  EXPECT_EQ(prod::static_name(), "m * s");
  static_assert(prod::static_dimension() == kul::length() * kul::time(),
      "static_product dimension");
}

TEST(static_product, push_back)
{
  using a = kul::static_product<>;
  using b = kul::push_back_t<a, kul::meter>;
  static_assert(std::is_same_v<b, kul::static_product<kul::meter>>,
      "first push_back");
  using c = kul::push_back_t<b, kul::second>;
  static_assert(std::is_same_v<c, kul::static_product<kul::meter, kul::second>>,
      "second push_back");
  using d = kul::push_back_t<c, kul::unitless>;
  static_assert(std::is_same_v<d, kul::static_product<kul::meter, kul::second, kul::unitless>>,
      "third push_back");
}

TEST(static_product, push_back_unless_unitless)
{
  using a = kul::static_product<>;
  using b = kul::push_back_unless_unitless_t<a, kul::meter>;
  static_assert(std::is_same_v<b, kul::static_product<kul::meter>>,
      "first push_back");
  using c = kul::push_back_unless_unitless_t<b, kul::unitless>;
  static_assert(std::is_same_v<c, kul::static_product<kul::meter>>,
      "third push_back");
}

TEST(static_product, multiply_with_exp)
{
  using a = kul::static_product<>;
  using b = kul::multiply_with_t<a, kul::static_pow<kul::meter, 1>>;
  static_assert(std::is_same_v<b, kul::static_product<kul::static_pow<kul::meter, 1>>>,
      "first push_back");
  using c = kul::multiply_with_t<b, kul::static_pow<kul::second, -1>>;
  static_assert(std::is_same_v<c, kul::static_product<kul::static_pow<kul::meter, 1>, kul::static_pow<kul::second, -1>>>,
      "second push_back");
  using d = kul::multiply_with_t<c, kul::static_pow<kul::second, -1>>;
  static_assert(std::is_same_v<d, kul::static_product<kul::static_pow<kul::meter, 1>, kul::static_pow<kul::second, -2>>>,
      "third push_back");
}

TEST(static_product, multiply_two_products)
{
  using a = kul::static_product<kul::static_pow<kul::meter, 1>, kul::static_pow<kul::second, -1>>;
  using b = kul::multiply_with_t<a, a>;
  using expected = kul::static_product<kul::static_pow<kul::meter, 2>, kul::static_pow<kul::second, -2>>;
  static_assert(std::is_same_v<b, expected>, "m/s * m/s = m^2/s^2");
}

TEST(static_product, multiply_with_named)
{
  using a = kul::static_product<kul::static_pow<kul::meter, 1>>;
  using b = kul::multiply_with_t<a, kul::meter>;
  using expected = kul::static_product<kul::static_pow<kul::meter, 2>>;
  static_assert(std::is_same_v<b, expected>, "multiplying with named unit");
}

TEST(static_product, divide_by)
{
  using a = kul::static_product<kul::static_pow<kul::meter, 1>, kul::static_pow<kul::second, -1>>;
  using b = kul::static_product<kul::static_pow<kul::meter, 1>>;
  using c = kul::divide_by_t<a, b>;
  using expected = kul::static_product<kul::static_pow<kul::meter, 0>, kul::static_pow<kul::second, -1>>;
  static_assert(std::is_same_v<c, expected>, "raw static division");
}

TEST(simplify, zero_exp)
{
  using a = kul::simplify_t<kul::static_pow<kul::meter, 0>>;
  static_assert(std::is_same_v<a, kul::unitless>, "zero exponent is unitless");
}

TEST(simplify, one_exp)
{
  using a = kul::simplify_t<kul::static_pow<kul::meter, 1>>;
  static_assert(std::is_same_v<a, kul::meter>, "one exponent is base");
}

TEST(simplify, terms)
{
  using a = kul::static_pow<kul::meter, 1>;
  using b = kul::static_pow<kul::second, -1>;
  using c = kul::static_product<a, b>;
  using d = kul::simplify_t<c>;
  using expected = kul::static_product<kul::meter, b>;
  static_assert(std::is_same_v<d, expected>, "simplify terms");
}

TEST(simplify, product)
{
  using a = kul::static_pow<kul::meter, 3>;
  using b = kul::static_pow<kul::second, 0>;
  using c = kul::static_product<a, b>;
  using d = kul::simplify_t<c>;
  static_assert(std::is_same_v<d, a>, "simplify product");
}

TEST(simplify, unitless)
{
  using a = kul::static_pow<kul::meter, 0>;
  using b = kul::static_pow<kul::second, 0>;
  using c = kul::static_product<a, b>;
  using d = kul::simplify_t<c>;
  static_assert(std::is_same_v<d, kul::unitless>, "simplify unitless");
}

TEST(static_multiply, square_meter)
{
  using a = kul::multiply<kul::meter, kul::meter>;
  using b = kul::static_pow<kul::meter, 2>;
  static_assert(std::is_same_v<a, b>, "square meter");
}

TEST(static_divide, m_per_s)
{
  using a = kul::divide<kul::meter, kul::second>;
  using b = kul::static_product<kul::meter, kul::static_pow<kul::second, -1>>;
  static_assert(std::is_same_v<a, b>, "m/s");
}

TEST(static_divide, m_per_m)
{
  using a = kul::divide<kul::meter, kul::meter>;
  static_assert(std::is_same_v<a, kul::unitless>, "m/m");
}

TEST(root, m3)
{
  auto a = kul::meter() * kul::meter() * kul::meter();
  auto b = kul::root(a, 3);
  EXPECT_EQ(b.name(), "m");
}

TEST(static_root, m3)
{
  using a = kul::static_pow<kul::meter, 3>;
  using b = kul::static_root<a, 3>;
  static_assert(std::is_same_v<b, kul::meter>, "static_root(m^3, 3)");
}

TEST(conversion, in_mm)
{
  auto c = kul::conversion<double>(kul::inch(), kul::milli<kul::meter>());
  auto v = c(1.0);
  EXPECT_EQ(v, 25.4);
}

TEST(static_conversion, in_mm)
{
  auto constexpr c = kul::static_conversion<double, kul::inch, kul::milli<kul::meter>>;
  auto constexpr v = c(1.0);
  static_assert(v == 25.4, "static inch to mm");
}

TEST(quantity, construct)
{
  auto constexpr b = kul::quantity<double, kul::meter>(2.0);
  static_assert(b.value() == 2.0, "constexpr quantity construction");
  auto const a = kul::quantity<double, kul::meter>(2.0);
  EXPECT_EQ(a.value(), 2.0);
}

TEST(quantity, compare)
{
  auto constexpr c1 = kul::quantity<double, kul::meter>(1.0);
  auto constexpr c2 = kul::quantity<double, kul::meter>(2.0);
  static_assert(c1 == c1, "constexpr equality");
  static_assert(!(c1 != c1), "constexpr non-equality");
  static_assert(!(c1 == c2), "constexpr equality of unequal");
  static_assert(c1 != c2, "constexpr non-equality of unequal");
  static_assert(c1 < c2, "constexpr less-than");
  static_assert(!(c1 > c2), "constexpr not greater-than");
  static_assert(!(c2 < c1), "constexpr not less-than");
  static_assert(c2 > c1, "constexpr greater-than");
  static_assert(c1 <= c2, "constexpr less-than-or-equal");
  static_assert(!(c1 >= c2), "constexpr not greater-than-or-equal");
  static_assert(!(c2 <= c1), "constexpr not less-than-or-equal");
  static_assert(c2 >= c1, "constexpr greater-than-or-equal");
  static_assert(c1 <= c1, "constexpr less-than-or-equal of equal");
  static_assert(c1 >= c1, "constexpr greater-than-or-equal of equal");
  auto const r1 = kul::quantity<double, kul::meter>(1.0);
  auto const r2 = kul::quantity<double, kul::meter>(2.0);
  EXPECT_TRUE(c1 == c1);
  EXPECT_TRUE(!(c1 != c1));
  EXPECT_TRUE(!(c1 == c2));
  EXPECT_TRUE(c1 != c2);
  EXPECT_TRUE(c1 < c2);
  EXPECT_TRUE(!(c1 > c2));
  EXPECT_TRUE(!(c2 < c1));
  EXPECT_TRUE(c2 > c1);
  EXPECT_TRUE(c1 <= c2);
  EXPECT_TRUE(!(c1 >= c2));
  EXPECT_TRUE(!(c2 <= c1));
  EXPECT_TRUE(c2 >= c1);
  EXPECT_TRUE(c1 <= c1);
  EXPECT_TRUE(c1 >= c1);
}

TEST(quantity, add)
{
  auto constexpr c1 = kul::quantity<double, kul::meter>(2.0);
  auto constexpr c2 = kul::quantity<double, kul::meter>(4.0);
  static_assert((c1 + c1) == c2, "constexpr quantity add");
  auto const r1 = c1;
  auto const r2 = c2;
  EXPECT_EQ(r1 + r1, r2);
}

TEST(quantity, subtract)
{
  auto constexpr c1 = kul::quantity<double, kul::meter>(2.0);
  auto constexpr c2 = kul::quantity<double, kul::meter>(1.0);
  static_assert((c1 - c2) == c2, "constexpr quantity subtract");
  auto const r1 = c1;
  auto const r2 = c2;
  EXPECT_EQ(r1 - r2, r2);
}

TEST(quantity, subtract_absolute)
{
  auto constexpr c1 = kul::kelvins<double>(2.0);
  auto constexpr c2 = kul::kelvins<double>(1.0);
  static_assert(c1.unit_origin().has_value(), "c1 is absolute");
  static_assert(c2.unit_origin().has_value(), "c2 is absolute");
  auto constexpr c3 = c1 - c2;
  static_assert(!c3.unit_origin().has_value(), "c3 is relative");
}

TEST(quantity, multiply)
{
  auto constexpr c1 = kul::quantity<double, kul::meter>(2.0);
  auto constexpr c2 = kul::quantity<double, kul::static_pow<kul::meter, 2>>(4.0);
  static_assert((c1 * c1) == c2, "constexpr quantity multiply");
  auto const r1 = c1;
  auto const r2 = c2;
  EXPECT_EQ((r1 * r1), r2);
}

TEST(quantity, divide)
{
  auto constexpr c1 = kul::quantity<double, kul::meter>(6.0);
  auto constexpr c2 = kul::quantity<double, kul::second>(2.0);
  auto constexpr c3 = kul::quantity<double,
       kul::static_product<kul::meter, kul::static_pow<kul::second, -1>>>(3.0);
  static_assert((c1 / c2) == c3, "constexpr quantity divide");
  auto const r1 = c1;
  auto const r2 = c2;
  auto const r3 = c3;
  EXPECT_EQ((r1 / r2), r3);
}

TEST(quantity, sqrt)
{
  auto const r1 = kul::quantity<double, kul::meter>(2.0);
  auto const r2 = kul::quantity<double, kul::static_pow<kul::meter, 2>>(4.0);
  EXPECT_EQ(kul::sqrt(r2), r1);
}

TEST(quantity, cbrt)
{
  auto const r1 = kul::quantity<double, kul::meter>(2.0);
  auto const r2 = kul::quantity<double, kul::static_pow<kul::meter, 3>>(8.0);
  EXPECT_EQ(kul::cbrt(r2), r1);
}

TEST(quantity, sin)
{
  auto r1 = kul::quantity<double, kul::radian>(0.0);
  auto r2 = kul::quantity<double, kul::unitless>(0.0);
  auto r3 = kul::sin(r1);
  static_assert(std::is_same_v<decltype(r3), decltype(r2)>,
      "sin returns unitless");
  EXPECT_EQ(r3, r2);
  auto r4 = kul::asin(r3);
  static_assert(std::is_same_v<decltype(r4), decltype(r1)>,
      "asin returns radians");
  EXPECT_EQ(r4, r1);
}

TEST(quantity, cos)
{
  auto r1 = kul::quantity<double, kul::radian>(0.0);
  auto r2 = kul::quantity<double, kul::unitless>(1.0);
  auto r3 = kul::cos(r1);
  static_assert(std::is_same_v<decltype(r3), decltype(r2)>,
      "cos returns unitless");
  EXPECT_EQ(r3, r2);
  auto r4 = kul::acos(r3);
  static_assert(std::is_same_v<decltype(r4), decltype(r1)>,
      "acos returns radians");
  EXPECT_EQ(r4, r1);
}

TEST(quantity, hypot)
{
  auto r1 = kul::quantity<double, kul::meter>(1.0);
  auto r2 = kul::quantity<double, kul::meter>(Kokkos::sqrt(3.0));
  auto r3 = kul::hypot(r1, r1, r1);
  static_assert(std::is_same_v<decltype(r3), decltype(r2)>,
      "hypot(meters) -> meters");
  EXPECT_FLOAT_EQ(r3.value(), r2.value());
}

TEST(quantity, fma)
{
  auto r1 = kul::quantity<double, kul::meter>(2.0);
  auto r2 = kul::quantity<double, kul::static_pow<kul::second, -1>>(3.0);
  auto r3 = kul::quantity<double, kul::static_product<kul::meter, kul::static_pow<kul::second, -1>>>(4.0);
  auto r4 = kul::quantity<double, kul::static_product<kul::meter, kul::static_pow<kul::second, -1>>>(10.0);
  auto r5 = kul::fma(r1, r2, r3);
  static_assert(std::is_same_v<decltype(r5), decltype(r4)>,
      "fma return type");
  EXPECT_FLOAT_EQ(r5.value(), r4.value());
}
