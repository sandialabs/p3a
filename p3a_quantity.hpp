#pragma once

#include "p3a_macros.hpp"
#include "p3a_unit.hpp"

namespace p3a {

/* This class represents, as a C++ object (not just a type),
 * a physical quantity.
 *
 * The first template argument, Unit, should be an instance of
 * the class template p3a::unit describing the unit of measurement
 * in which the quantity is represented.
 *
 * The second template argument, ValueType, should be the type
 * used to store the numerical value of the quantity
 * as a factor to multiply by the unit of measure.
 * ValueType is most often a floating-point type of some sort.
 *
 * The third template argument handles the subtle distinction between
 * relative and absolute quantities, for example the difference between
 * a point in space versus a vector in space, or the difference between
 * an absolute temperature versus a change in temperature.
 * Mathematically, quantities can be thought of as forming an affine space
 * with points and translation vectors.
 * In P3A, if Origin is void, then this quantity is a "vector quantity",
 * which means a relative quantity such as a displacement.
 * If Origin is an instance of std::ratio, then this quantity is a
 * "point quantity", which means an absolute quantity such as a position.
 * The actual rational number represented by the std::ratio expresses a
 * compile-time origin, measured in the SI unit used to measure quantities
 * of this dimension.
 * This is useful, for example, to describe how Kelvin, Celcius, and
 * Fahrenheit scales have different definitions of zero.
 */

template <class Unit, class ValueType = double, class Origin = void>
class quantity {
 public:
  using value_type = ValueType;
  using unit = Unit;
  using dimension = typename unit::dimension;
  using unit_magnitude = typename unit::magnitude;
  using origin = Origin;
 private:
  value_type m_value;
 public:
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  explicit quantity(value_type const& r)
    :m_value(r)
  {}
  P3A_ALWAYS_INLINE inline
  quantity() = default;
  P3A_ALWAYS_INLINE inline constexpr
  quantity(quantity const&) = default;
  P3A_ALWAYS_INLINE inline constexpr
  quantity(quantity&&) = default;
  P3A_ALWAYS_INLINE inline constexpr
  quantity& operator=(quantity const&) = default;
  P3A_ALWAYS_INLINE inline constexpr
  quantity& operator=(quantity&&) = default;
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  value_type const& value() const { return m_value; }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  value_type& value() { return m_value; }
  // converting constructor for converting only ValueType
  template <class OtherValueType>
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  explicit quantity(quantity<unit, OtherValueType, origin> const& other)
    :m_value(other.value())
  {
  }
  // converting constructor for different magnitude and/or origin
  template <class OtherUnit, class OtherValueType, class OtherOrigin>
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  explicit quantity(quantity<OtherUnit, OtherValueType, OtherOrigin> const& other)
    :m_value(0)
  {
    using other_dimension = typename OtherUnit::dimension;
    using other_unit_magnitude = typename OtherUnit::magnitude;
    static_assert(std::is_same_v<dimension, other_dimension>,
        "not allowed to convert between quantities of different dimensions");
    OtherValueType other_si_value =
      other.value() * OtherValueType(other_unit_magnitude::num)
      / OtherValueType(other_unit_magnitude::den);
    if constexpr (!std::is_same_v<OtherOrigin, void>) {
      other_si_value += OtherValueType(OtherOrigin::num)
        / OtherValueType(OtherOrigin::den);
    }
    value_type si_value = value_type(other_si_value);
    if constexpr (!std::is_same_v<origin, void>) {
      si_value -= value_type(origin::num)
        / value_type(origin::den);
    }
    m_value = si_value * value_type(unit_magnitude::den)
      / value_type(unit_magnitude::num);
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  auto operator==(quantity const& other) const {
    return value() == other.value();
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  auto operator!=(quantity const& other) const {
    return value() != other.value();
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  auto operator<=(quantity const& other) const {
    return value() <= other.value();
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  auto operator>=(quantity const& other) const {
    return value() >= other.value();
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  auto operator<(quantity const& other) const {
    return value() < other.value();
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  auto operator>(quantity const& other) const {
    return value() > other.value();
  }
};

template <class ValueType>
class quantity<no_unit, ValueType, void> {
 public:
  using value_type = ValueType;
  using unit = no_unit;
  using dimension = typename unit::dimension;
  using unit_magnitude = typename unit::magnitude;
  using origin = void;
 private:
  value_type m_value;
 public:
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  quantity(value_type const& r)
    :m_value(r)
  {}
  P3A_ALWAYS_INLINE inline
  quantity() = default;
  P3A_ALWAYS_INLINE inline constexpr
  quantity(quantity const&) = default;
  P3A_ALWAYS_INLINE inline constexpr
  quantity(quantity&&) = default;
  P3A_ALWAYS_INLINE inline constexpr
  quantity& operator=(quantity const&) = default;
  P3A_ALWAYS_INLINE inline constexpr
  quantity& operator=(quantity&&) = default;
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  value_type const& value() const { return m_value; }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  value_type& value() { return m_value; }
  // converting constructor for converting only ValueType
  template <class OtherValueType>
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  explicit quantity(quantity<unit, OtherValueType, origin> const& other)
    :m_value(other.value())
  {
  }
  // converting constructor for different magnitude and/or origin
  template <class OtherUnit, class OtherValueType, class OtherOrigin>
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  explicit quantity(quantity<OtherUnit, OtherValueType, OtherOrigin> const& other)
    :m_value(0)
  {
    using other_dimension = typename OtherUnit::dimension;
    using other_unit_magnitude = typename OtherUnit::magnitude;
    static_assert(std::is_same_v<dimension, other_dimension>,
        "not allowed to convert between quantities of different dimensions");
    OtherValueType other_si_value =
      other.value() * OtherValueType(other_unit_magnitude::num)
      / OtherValueType(other_unit_magnitude::den);
    if constexpr (!std::is_same_v<OtherOrigin, void>) {
      other_si_value += OtherValueType(OtherOrigin::num)
        / OtherValueType(OtherOrigin::den);
    }
    value_type si_value = value_type(other_si_value);
    m_value = si_value * value_type(unit_magnitude::den)
      / value_type(unit_magnitude::num);
  }
};

template <class Unit, class ValueType = double>
using absolute_quantity = quantity<Unit, ValueType, std::ratio<0>>;

// some common quantities

template <class ValueType = double>
using unitless = quantity<no_unit, ValueType>;

template <class ValueType = double>
using seconds = quantity<second, ValueType>;
template <class ValueType = double>
using milliseconds = quantity<millisecond, ValueType>;
template <class ValueType = double>
using microseconds = quantity<microsecond, ValueType>;
template <class ValueType = double>
using nanoseconds = quantity<nanosecond, ValueType>;

template <class ValueType = double>
using meters = quantity<meter, ValueType>;
template <class ValueType = double>
using centimeters = quantity<centimeter, ValueType>;
template <class ValueType = double>
using millimeters = quantity<millimeter, ValueType>;
template <class ValueType = double>
using micrometers = quantity<micrometer, ValueType>;
template <class ValueType = double>
using nanometers = quantity<nanometer, ValueType>;

template <class ValueType = double>
using grams = quantity<gram, ValueType>;
template <class ValueType = double>
using kilograms = quantity<kilogram, ValueType>;
template <class ValueType = double>
using amperes = quantity<ampere, ValueType>;

// temperature units are, by default, absolute units to describe their
// relative zero offsets

template <class ValueType = double>
using degrees_kelvin = absolute_quantity<degree_kelvin, ValueType>;
template <class ValueType = double>
using degrees_celcius = quantity<degree_celcius, ValueType, std::ratio<27315, 100>>;
template <class ValueType = double>
using degrees_fahrenheit = quantity<degree_fahrenheit, ValueType,
      std::ratio_multiply<std::ratio<5, 9>, std::ratio<45967, 100>>>;
template <class ValueType = double>
using temperature_electronvolts =
  absolute_quantity<temperature_electronvolt, ValueType>;

template <class ValueType = double>
using meters_per_second = quantity<meter_per_second, ValueType>;
template <class ValueType = double>
using meters_per_second_squared = quantity<meter_per_second_squared, ValueType>;

template <class ValueType = double>
using square_meters = quantity<square_meter, ValueType>;
template <class ValueType = double>
using cubic_meters = quantity<cubic_meter, ValueType>;

template <class ValueType = double>
using kilograms_per_cubic_meter = quantity<kilogram_per_cubic_meter, ValueType>;
template <class ValueType = double>
using grams_per_cubic_centimeter = quantity<gram_per_cubic_centimeter, ValueType>;

template <class ValueType = double>
using joules = quantity<joule, ValueType>;
template <class ValueType = double>
using watts = quantity<watt, ValueType>;
template <class ValueType = double>
using joules_per_cubic_meter = quantity<joule_per_cubic_meter, ValueType>;
template <class ValueType = double>
using joules_per_kilogram = quantity<joule_per_kilogram, ValueType>;

template <class ValueType = double>
using newtons = quantity<newton, ValueType>;
template <class ValueType = double>
using pascals = quantity<pascal, ValueType>;

template <class ValueType = double>
using arc_degrees = quantity<arc_degree, ValueType>;
template <class ValueType = double>
using percentage = quantity<percent, ValueType>;

// four arithmetic binary operators

template <
  class LeftUnit,
  class LeftValueType,
  class LeftOrigin,
  class RightUnit,
  class RightValueType,
  class RightOrigin>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
auto operator+(
    quantity<LeftUnit, LeftValueType, LeftOrigin> const& left,
    quantity<RightUnit, RightValueType, RightOrigin> const& right)
{
  using left_dimension = typename LeftUnit::dimension;
  using right_dimension = typename RightUnit::dimension;
  using left_magnitude = typename LeftUnit::magnitude;
  using right_magnitude = typename RightUnit::magnitude;
  static_assert(std::is_same_v<left_dimension, right_dimension>,
      "not allowed to add quantities with different dimensions");
  static_assert(std::is_same_v<left_magnitude, right_magnitude>,
      "not allowed to add quantities with the same dimension but different units");
  bool constexpr left_is_relative = std::is_same_v<LeftOrigin, void>;
  bool constexpr right_is_relative = std::is_same_v<RightOrigin, void>;
  bool constexpr left_is_absolute = !left_is_relative;
  bool constexpr right_is_absolute = !right_is_relative;
  static_assert(!(left_is_absolute && right_is_absolute),
      "not allowed to add two absolute (affine point) quantities");
  using origin = std::conditional_t<
    (left_is_relative && right_is_relative),
    void,
    std::conditional_t<
    left_is_absolute,
    LeftOrigin,
    RightOrigin>>;
  auto const value = left.value() + right.value();
  using value_type = std::remove_const_t<decltype(value)>;
  return quantity<LeftUnit, value_type, origin>(value);
}

template <
  class LeftUnit,
  class LeftValueType,
  class LeftOrigin,
  class RightUnit,
  class RightValueType,
  class RightOrigin>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
auto operator-(
    quantity<LeftUnit, LeftValueType, LeftOrigin> const& left,
    quantity<RightUnit, RightValueType, RightOrigin> const& right)
{
  using left_dimension = typename LeftUnit::dimension;
  using right_dimension = typename RightUnit::dimension;
  using left_magnitude = typename LeftUnit::magnitude;
  using right_magnitude = typename RightUnit::magnitude;
  static_assert(std::is_same_v<left_dimension, right_dimension>,
      "not allowed to subtract quantities with different dimensions");
  static_assert(std::is_same_v<left_magnitude, right_magnitude>,
      "not allowed to subtract quantities with the same dimension but different units");
  bool constexpr left_is_relative = std::is_same_v<LeftOrigin, void>;
  bool constexpr right_is_relative = std::is_same_v<RightOrigin, void>;
  bool constexpr left_is_absolute = !left_is_relative;
  bool constexpr right_is_absolute = !right_is_relative;
  static_assert(!(left_is_relative && right_is_absolute),
      "not allowed to subtract an absolute (affine point) quantity from a relative (affine vector) one");
  if constexpr (left_is_absolute && right_is_absolute) {
    static_assert(std::is_same_v<LeftOrigin, RightOrigin>,
        "not allowed to subtract two absolute (affine point) quantities with different origins");
  }
  using origin = std::conditional_t<
    (left_is_absolute && right_is_relative),
    LeftOrigin,
    void>;
  auto const value = left.value() - right.value();
  using value_type = std::remove_const_t<decltype(value)>;
  return quantity<LeftUnit, value_type, origin>(value);
}

template <
  class LeftUnit,
  class LeftValueType,
  class LeftOrigin,
  class RightUnit,
  class RightValueType,
  class RightOrigin>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
auto operator*(
    quantity<LeftUnit, LeftValueType, LeftOrigin> const& left,
    quantity<RightUnit, RightValueType, RightOrigin> const& right)
{
  using result_unit = unit_multiply<LeftUnit, RightUnit>;
  bool constexpr left_is_relative = std::is_same_v<LeftOrigin, void>;
  bool constexpr right_is_relative = std::is_same_v<RightOrigin, void>;
  static_assert((left_is_relative && right_is_relative),
      "not allowed to multiply absolute (affine point) quantities by other quantities");
  auto const value = left.value() * right.value();
  using value_type = std::remove_const_t<decltype(value)>;
  return quantity<result_unit, value_type, void>(value);
}

template <
  class LeftUnit,
  class LeftValueType,
  class LeftOrigin,
  class RightUnit,
  class RightValueType,
  class RightOrigin>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
auto operator/(
    quantity<LeftUnit, LeftValueType, LeftOrigin> const& left,
    quantity<RightUnit, RightValueType, RightOrigin> const& right)
{
  using result_unit = unit_divide<LeftUnit, RightUnit>;
  bool constexpr left_is_relative = std::is_same_v<LeftOrigin, void>;
  bool constexpr right_is_relative = std::is_same_v<RightOrigin, void>;
  static_assert((left_is_relative && right_is_relative),
      "not allowed to divide absolute (affine point) quantities by other quantities");
  auto const value = left.value() / right.value();
  using value_type = std::remove_const_t<decltype(value)>;
  return quantity<result_unit, value_type, void>(value);
}

// combined assignment versions of four arithmetic operators

template <
  class LeftUnit,
  class LeftValueType,
  class LeftOrigin,
  class RightUnit,
  class RightValueType,
  class RightOrigin>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
quantity<LeftUnit, LeftValueType, LeftOrigin>&
operator+=(
    quantity<LeftUnit, LeftValueType, LeftOrigin>& left,
    quantity<RightUnit, RightValueType, RightOrigin> const& right)
{
  left = left + right;
  return left;
}

template <
  class LeftUnit,
  class LeftValueType,
  class LeftOrigin,
  class RightUnit,
  class RightValueType,
  class RightOrigin>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
quantity<LeftUnit, LeftValueType, LeftOrigin>&
operator-=(
    quantity<LeftUnit, LeftValueType, LeftOrigin>& left,
    quantity<RightUnit, RightValueType, RightOrigin> const& right)
{
  left = left - right;
  return left;
}

template <
  class LeftUnit,
  class LeftValueType,
  class LeftOrigin,
  class RightUnit,
  class RightValueType,
  class RightOrigin>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
quantity<LeftUnit, LeftValueType, LeftOrigin>&
operator*=(
    quantity<LeftUnit, LeftValueType, LeftOrigin>& left,
    quantity<RightUnit, RightValueType, RightOrigin> const& right)
{
  left = left * right;
  return left;
}

template <
  class LeftUnit,
  class LeftValueType,
  class LeftOrigin,
  class RightUnit,
  class RightValueType,
  class RightOrigin>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
quantity<LeftUnit, LeftValueType, LeftOrigin>&
operator/=(
    quantity<LeftUnit, LeftValueType, LeftOrigin>& left,
    quantity<RightUnit, RightValueType, RightOrigin> const& right)
{
  left = left / right;
  return left;
}

// build a quantity by multiplying a number by a unit

template <
  class ValueType,
  class Dimension,
  class Magnitude>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
std::enable_if_t<std::is_arithmetic_v<ValueType>,
  quantity<unit<Dimension, Magnitude>, ValueType, void>>
operator*(
    ValueType const& left,
    unit<Dimension, Magnitude>)
{
  return quantity<unit<Dimension, Magnitude>, ValueType, void>(left);
}

}
