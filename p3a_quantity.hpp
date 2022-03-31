#pragma once

#include "p3a_macros.hpp"
#include "p3a_unit.hpp"
#include "p3a_scalar.hpp"

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
  // constructor from a plain arithmetic type is explicit for non-unitless quantities
  template <class T,
      typename std::enable_if<
         (!std::is_same_v<quantity<Unit, ValueType, Origin>, quantity<no_unit, ValueType, void>>) &&
         std::is_arithmetic_v<T>,
         bool>::type = false>
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  explicit quantity(T const& v)
    :m_value(v)
  {}
  // constructor from a plain arithmetic type is implicit for unitless quantities
  template <class T,
      typename std::enable_if<
         std::is_same_v<quantity<Unit, ValueType, Origin>, quantity<no_unit, ValueType, void>> &&
         std::is_arithmetic_v<T>,
         bool>::type = false>
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  quantity(T const& v)
    :m_value(v)
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
      static_assert(!std::is_same_v<origin, void>,
          "not allowed to convert from an absolute quantity to a relative one");
      other_si_value += OtherValueType(OtherOrigin::num)
        / OtherValueType(OtherOrigin::den);
    }
    value_type si_value = value_type(other_si_value);
    if constexpr (!std::is_same_v<origin, void>) {
      static_assert(!std::is_same_v<OtherOrigin, void>,
          "not allowed to convert from a relative quantity to an absolute one");
      si_value -= value_type(origin::num)
        / value_type(origin::den);
    }
    m_value = si_value * value_type(unit_magnitude::den)
      / value_type(unit_magnitude::num);
  }
  // converting assignment operator
  template <class OtherUnit, class OtherValueType, class OtherOrigin>
  P3A_ALWAYS_INLINE inline constexpr
  quantity& operator=(quantity<OtherUnit, OtherValueType, OtherOrigin> const& other) {
    // attempt converting constructor then regular assign
    return operator=(quantity<unit, value_type, origin>(other));
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

namespace details {

template <class Unit, class ValueType, class Origin>
struct is_scalar<quantity<Unit, ValueType, Origin>> {
  inline static constexpr bool value = true;
};

}

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
using megajoules_per_kilogram = quantity<megajoule_per_kilogram, ValueType>;
template <class ValueType = double>
using joules_per_kilogram_per_kelvin = quantity<joule_per_kilogram_per_kelvin, ValueType>;

template <class ValueType = double>
using newtons = quantity<newton, ValueType>;
template <class ValueType = double>
using pascals = quantity<pascal, ValueType>;
template <class ValueType = double>
using gigapascals = quantity<gigapascal, ValueType>;

template <class ValueType = double>
using arc_degrees = quantity<arc_degree, ValueType>;
template <class ValueType = double>
using percentage = quantity<percent, ValueType>;

template <class ValueType = double>
using siemens_quantity = quantity<siemens, ValueType>;
template <class ValueType = double>
using siemens_per_meter_quantity = quantity<siemens_per_meter, ValueType>;

template <class ValueType = double>
using gaussian_electrical_conductivity_quantity = quantity<gaussian_electrical_conductivity_unit, ValueType>;

template <
  class Unit,
  class ValueType,
  class Origin>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
quantity<Unit, ValueType, Origin> operator-(quantity<Unit, ValueType, Origin> const& q)
{
  static_assert(std::is_same_v<Origin, void>,
      "not allowed to negate absolute quantities");
  return quantity<Unit, ValueType, Origin>(-(q.value()));
}

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
  class Right>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
quantity<LeftUnit, LeftValueType, LeftOrigin>&
operator+=(
    quantity<LeftUnit, LeftValueType, LeftOrigin>& left,
    Right const& right)
{
  left = left + right;
  return left;
}

template <
  class LeftUnit,
  class LeftValueType,
  class LeftOrigin,
  class Right>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
quantity<LeftUnit, LeftValueType, LeftOrigin>&
operator-=(
    quantity<LeftUnit, LeftValueType, LeftOrigin>& left,
    Right const& right)
{
  left = left - right;
  return left;
}

template <
  class LeftUnit,
  class LeftValueType,
  class LeftOrigin,
  class Right>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
quantity<LeftUnit, LeftValueType, LeftOrigin>&
operator*=(
    quantity<LeftUnit, LeftValueType, LeftOrigin>& left,
    Right const& right)
{
  left = left * right;
  return left;
}

template <
  class LeftUnit,
  class LeftValueType,
  class LeftOrigin,
  class Right>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
quantity<LeftUnit, LeftValueType, LeftOrigin>&
operator/=(
    quantity<LeftUnit, LeftValueType, LeftOrigin>& left,
    Right const& right)
{
  left = left / right;
  return left;
}

// build a quantity by multiplying a number by a unit

template <
  class Arithmetic,
  class Dimension,
  class Magnitude>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
std::enable_if_t<std::is_arithmetic_v<Arithmetic>,
  quantity<unit<Dimension, Magnitude>, Arithmetic, void>>
operator*(
    Arithmetic const& left,
    unit<Dimension, Magnitude>)
{
  return quantity<unit<Dimension, Magnitude>, Arithmetic, void>(left);
}

// binary math operators that promote an operand of a built-in arithmetic
// type into a unitless quantity, as long as the other operand is already
// a physical quantity

template <class Arithmetic, class Unit, class ValueType, class Origin>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
std::enable_if_t<std::is_arithmetic_v<Arithmetic>, quantity<Unit, ValueType, Origin>>
operator+(
    Arithmetic const& left,
    quantity<Unit, ValueType, Origin> const& right)
{
  return unitless<Arithmetic>(left) + right;
}

template <class Arithmetic, class Unit, class ValueType, class Origin>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
std::enable_if_t<std::is_arithmetic_v<Arithmetic>, quantity<Unit, ValueType, Origin>>
operator+(
    quantity<Unit, ValueType, Origin> const& left,
    Arithmetic const& right)
{
  return left + unitless<Arithmetic>(right);
}

template <class Arithmetic, class Unit, class ValueType, class Origin>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
std::enable_if_t<std::is_arithmetic_v<Arithmetic>, quantity<Unit, ValueType, Origin>>
operator-(
    Arithmetic const& left,
    quantity<Unit, ValueType, Origin> const& right)
{
  return unitless<Arithmetic>(left) - right;
}

template <class Arithmetic, class Unit, class ValueType, class Origin>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
std::enable_if_t<std::is_arithmetic_v<Arithmetic>, quantity<Unit, ValueType, Origin>>
operator-(
    quantity<Unit, ValueType, Origin> const& left,
    Arithmetic const& right)
{
  return left - unitless<Arithmetic>(right);
}

template <class Arithmetic, class Unit, class ValueType, class Origin>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
std::enable_if_t<std::is_arithmetic_v<Arithmetic>, quantity<Unit, ValueType, Origin>>
operator*(
    Arithmetic const& left,
    quantity<Unit, ValueType, Origin> const& right)
{
  return unitless<Arithmetic>(left) * right;
}

template <class Arithmetic, class Unit, class ValueType, class Origin>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
std::enable_if_t<std::is_arithmetic_v<Arithmetic>, quantity<Unit, ValueType, Origin>>
operator*(
    quantity<Unit, ValueType, Origin> const& left,
    Arithmetic const& right)
{
  return left * unitless<Arithmetic>(right);
}

template <class Arithmetic, class Unit, class ValueType, class Origin>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
std::enable_if_t<std::is_arithmetic_v<Arithmetic>, quantity<unit_inverse<Unit>, ValueType, Origin>>
operator/(
    Arithmetic const& left,
    quantity<Unit, ValueType, Origin> const& right)
{
  return unitless<Arithmetic>(left) / right;
}

template <class Arithmetic, class Unit, class ValueType, class Origin>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
std::enable_if_t<std::is_arithmetic_v<Arithmetic>, quantity<Unit, ValueType, Origin>>
operator/(
    quantity<Unit, ValueType, Origin> const& left,
    Arithmetic const& right)
{
  return left / unitless<Arithmetic>(right);
}

// roots functions

template <class Unit, class ValueType, class Origin>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
auto square_root(quantity<Unit, ValueType, Origin> const& q)
{
  static_assert(std::is_same_v<Origin, void>,
      "not allowed to take square roots of absolute quantities");
  return quantity<unit_root<Unit, 2>, ValueType, Origin>(square_root(q.value()));
}

template <class Unit, class ValueType, class Origin>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
auto cube_root(quantity<Unit, ValueType, Origin> const& q)
{
  static_assert(std::is_same_v<Origin, void>,
      "not allowed to take cube roots of absolute quantities");
  return quantity<unit_root<Unit, 3>, ValueType, Origin>(cube_root(q.value()));
}

namespace quantity_literals {

// C++11 user-defined literals for common units
//
// to avoid polluting the namespace excessively, these have their own
// namespace and should be used by adding
//
// using namespace p3a::quantity_literals;
//
// to the relevant user function
//
// C++11 user-defined literals are quite hard to understand
// how to use, as far as I can see floating-point literals
// just have to use long double as the argument type

P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
p3a::seconds<double> operator""_s(long double v)
{
  return p3a::seconds<double>(v);
}

P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
p3a::meters<double> operator""_m(long double v)
{
  return p3a::meters<double>(v);
}

P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
p3a::kilograms<double> operator""_kg(long double v)
{
  return p3a::kilograms<double>(v);
}

P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
p3a::amperes<double> operator""_A(long double v)
{
  return p3a::amperes<double>(v);
}

P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
p3a::degrees_kelvin<double> operator""_K(long double v)
{
  return p3a::degrees_kelvin<double>(v);
}

P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
p3a::pascals<double> operator""_Pa(long double v)
{
  return p3a::pascals<double>(v);
}

P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
p3a::kilograms_per_cubic_meter<double> operator""_kg_per_m3(long double v)
{
  return p3a::kilograms_per_cubic_meter<double>(v);
}

P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
p3a::joules_per_kilogram<double> operator""_J_per_kg(long double v)
{
  return p3a::joules_per_kilogram<double>(v);
}

P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
p3a::siemens_per_meter_quantity<double> operator""_S_per_m(long double v)
{
  return p3a::siemens_per_meter_quantity<double>(v);
}

}

}
