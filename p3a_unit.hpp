#pragma once

#include <ratio>
#include <string>
#include <type_traits>

#include "p3a_dimension.hpp"

namespace p3a {

/* These classes represent a unit of measurement as a C++ type.
 *
 * The representation includes a physical dimension, represented by an instance of
 * the p3a::dimension class template, and a magnitude.
 * The magnitude should be an instance of the std::ratio class
 * template, which represents a rational number.
 * That rational number should be the magnitude of the unit
 * being described, as measured in the corresponding SI unit.
 * For example, if describing the kilometer unit of length,
 * the magnitude should be std::ratio<1000, 1>
 * because one kilometer equals 1000 meters and the meter
 * is the SI unit of length.
 * The classes also include static member methods name() to produce
 * a string description of the unit name.
 */

// Section 1: Basic Named Units: SI, Imperial, and more. Each gives dimension/magnitude/name

namespace details {

using pi_ratio = std::ratio<
  31415926535897932,
  10000000000000000>;

using speed_of_light_in_centimeters_per_second = std::ratio<29979245800>;

}

class no_unit {
 public:
  using dimension = no_dimension;
  using magnitude = std::ratio<1>;
  static std::string name() { return "1"; }
};

class radian {
 public:
  using dimension = no_dimension;
  using magnitude = std::ratio<1>;
  static std::string name() { return "rad"; }
};

class arc_degree {
 public:
  using dimension = no_dimension;
  using magnitude = std::ratio_divide<details::pi_ratio, std::ratio<180>>;
  static std::string name() { return "arcdegree"; }
};

class percent {
 public:
  using dimension = no_dimension;
  using magnitude = std::ratio<1, 100>;
  static std::string name() { return "percent"; }
};

class meter {
 public:
  using dimension = length;
  using magnitude = std::ratio<1>;
  static std::string name() { return "m"; }
};

class second {
 public:
  using dimension = time;
  using magnitude = std::ratio<1>;
  static std::string name() { return "s"; }
};

class inch {
 public:
  using dimension = length;
  using magnitude = std::ratio<254, 10'000>;
  static std::string name() { return "in"; }
};

class gram {
 public:
  using dimension = mass;
  using magnitude = std::ratio<1, 1000>;
  static std::string name() { return "g"; }
};

class ampere {
 public:
  using dimension = electric_current;
  using magnitude = std::ratio<1>;
  static std::string name() { return "A"; }
};

class degree_kelvin {
 public:
  using dimension = temperature;
  using magnitude = std::ratio<1>;
  static std::string name() { return "degreeK"; }
};

class degree_celcius {
 public:
  using dimension = temperature;
  using magnitude = std::ratio<1>;
  static std::string name() { return "degreeC"; }
};

class degree_fahrenheit {
 public:
  using dimension = temperature;
  using magnitude = std::ratio<5, 9>;
  static std::string name() { return "degreeF"; }
};

class temperature_electronvolt {
 public:
  using dimension = temperature;
  using magnitude = std::ratio<1160451812, 100000>;
  static std::string name() { return "eV"; }
};

class mole {
 public:
  using dimension = amount_of_substance;
  using magnitude = std::ratio<1>;
  static std::string name() { return "mol"; }
};

class candela {
 public:
  using dimension = amount_of_substance;
  using magnitude = std::ratio<1>;
  static std::string name() { return "cd"; }
};

class hertz {
 public:
  using dimension = frequency;
  using magnitude = std::ratio<1>;
  static std::string name() { return "Hz"; }
};

class joule {
 public:
  using dimension = energy;
  using magnitude = std::ratio<1>;
  static std::string name() { return "J"; }
};

class watt {
 public:
  using dimension = power;
  using magnitude = std::ratio<1>;
  static std::string name() { return "W"; }
};

class newton {
 public:
  using dimension = force;
  using magnitude = std::ratio<1>;
  static std::string name() { return "N"; }
};

class pascal {
 public:
  using dimension = pressure;
  using magnitude = std::ratio<1>;
  static std::string name() { return "Pa"; }
};

class volt {
 public:
  using dimension = electric_potential;
  using magnitude = std::ratio<1>;
  static std::string name() { return "V"; }
};

class siemens {
 public:
  using dimension = electrical_conductance;
  using magnitude = std::ratio<1>;
  static std::string name() { return "S"; }
};

class ohm {
 public:
  using dimension = electrical_resistance;
  using magnitude = std::ratio<1>;
  static std::string name() { return "Ohm"; }
};

class henry {
 public:
  using dimension = inductance;
  using magnitude = std::ratio<1>;
  static std::string name() { return "H"; }
};

class farad {
 public:
  using dimension = capacitance;
  using magnitude = std::ratio<1>;
  static std::string name() { return "F"; }
};

// The so-called "second" of electrical resistivity used in Gaussian units
// is equivalent to c^2 / 10^11 Ohm-meters
// where c is the speed of light in centimeters per second
// https://en.wikipedia.org/wiki/Centimetre%E2%80%93gram%E2%80%93second_system_of_units#Electromagnetic_units_in_various_CGS_systems

class gaussian_electrical_resistivity_unit {
 public:
  using dimension = electrical_resistivity;
  using magnitude =
    std::ratio_multiply<
      details::speed_of_light_in_centimeters_per_second,
      std::ratio_divide<
        details::speed_of_light_in_centimeters_per_second,
        std::ratio<100000000000>>>;
  static std::string name() { return "s"; }
};

// Section 2: Unit prefix templates: given a named unit above, applies a metric prefix

template <class Unit>
class nano {
 public:
  using dimension = typename Unit::dimension;
  using magnitude = std::ratio_divide<typename Unit::magnitude, std::ratio<1'000'000'000>>;
  static std::string name() { return "n" + Unit::name(); }
};

template <class Unit>
class micro {
 public:
  using dimension = typename Unit::dimension;
  using magnitude = std::ratio_divide<typename Unit::magnitude, std::ratio<1'000'000>>;
  static std::string name() { return "u" + Unit::name(); }
};

template <class Unit>
class milli {
 public:
  using dimension = typename Unit::dimension;
  using magnitude = std::ratio_divide<typename Unit::magnitude, std::ratio<1'000>>;
  static std::string name() { return "m" + Unit::name(); }
};

template <class Unit>
class centi {
 public:
  using dimension = typename Unit::dimension;
  using magnitude = std::ratio_divide<typename Unit::magnitude, std::ratio<100>>;
  static std::string name() { return "c" + Unit::name(); }
};

template <class Unit>
class kilo {
 public:
  using dimension = typename Unit::dimension;
  using magnitude = std::ratio_multiply<typename Unit::magnitude, std::ratio<1'000>>;
  static std::string name() { return "k" + Unit::name(); }
};

template <class Unit>
class mega {
 public:
  using dimension = typename Unit::dimension;
  using magnitude = std::ratio_multiply<typename Unit::magnitude, std::ratio<1'000'000>>;
  static std::string name() { return "M" + Unit::name(); }
};

template <class Unit>
class giga {
 public:
  using dimension = typename Unit::dimension;
  using magnitude = std::ratio_multiply<typename Unit::magnitude, std::ratio<1'000'000'000>>;
  static std::string name() { return "G" + Unit::name(); }
};

// Section 3: std::ratio extra functionality for exponents and roots (roots are only trivially supported) 

namespace details {

template <class Ratio, int Exponent, bool IsPositive = (Exponent > 0)>
class ratio_exp;

template <class Ratio>
class ratio_exp<Ratio, 0, false> {
 public:
  using type = std::ratio<1>;
};

template <class Ratio>
class ratio_exp<Ratio, 1, true> {
 public:
  using type = Ratio;
};

template <class Ratio, int Exponent>
class ratio_exp<Ratio, Exponent, true> {
 public:
  using type = std::ratio_multiply<typename ratio_exp<Ratio, Exponent - 1>::type, Ratio>;
};

template <class Ratio, int Exponent>
class ratio_exp<Ratio, Exponent, false> {
 public:
  using type = std::ratio_divide<typename ratio_exp<Ratio, Exponent + 1>::type, Ratio>;
};

template <class Ratio, int Root>
class ratio_root {
  static_assert(std::is_same_v<typename Ratio::type, std::ratio<1>>,
      "taking roots of std::ratio other than one is not supported yet"); 
 public:
  using type = std::ratio<1>;
};

}

template <class Ratio, int Exponent>
using ratio_exp = typename details::ratio_exp<Ratio, Exponent>::type;

template <class Ratio, int Root>
using ratio_root = typename details::ratio_root<Ratio, Root>::type;

// Section 4: class templates for named unit raised to power and product of named units

template <class Unit, int Exponent>
class unit_exp {
 public:
  using dimension = dimension_exp<typename Unit::dimension, Exponent>;
  using magnitude = ratio_exp<typename Unit::magnitude, Exponent>;
  static std::string name() { return Unit::name() + "^" + std::to_string(Exponent); }
};

template <class... Units>
class unit_product;


template <>
class unit_product<> {
 public:
  using dimension = typename no_unit::dimension;
  using magnitude = typename no_unit::magnitude;
  static std::string name() { return no_unit::name(); }
};

template <class LastUnit>
class unit_product<LastUnit> {
 public:
  using dimension = typename LastUnit::dimension;
  using magnitude = typename LastUnit::magnitude;
  static std::string name();
};

template <class FirstUnit, class... OtherUnits>
class unit_product<FirstUnit, OtherUnits...> {
 public:
  using dimension = dimension_multiply<
    typename FirstUnit::dimension, typename unit_product<OtherUnits...>::dimension>;
  using magnitude = std::ratio_multiply<
    typename FirstUnit::magnitude, typename unit_product<OtherUnits...>::magnitude>;
  static std::string name();
};


// Section 5: details helpers for manipulating products of units, supports basic math operations

namespace details {

template <class A>
class is_unit_product {
 public:
  static inline constexpr bool value = false;
};

template <class... Units>
class is_unit_product<unit_product<Units...>> {
 public:
  static inline constexpr bool value = true;
};

template <class A, class B>
class prepend_unit_product {
 public:
  using type = unit_product<A, B>;
};

template <class A>
class prepend_unit_product<A, no_unit> {
 public:
  using type = A;
};

template <class A, class... OtherUnits>
class prepend_unit_product<A, unit_product<OtherUnits...>> {
 public:
  using type = unit_product<A, OtherUnits...>;
};

template <class... OtherUnits>
class prepend_unit_product<no_unit, unit_product<OtherUnits...>> {
 public:
  using type = unit_product<OtherUnits...>;
};

template <class T>
class canonicalize_unit_exp {
 public:
  using type = unit_exp<T, 1>;
};

template <class T, int Exponent>
class canonicalize_unit_exp<unit_exp<T, Exponent>> {
 public:
  using type = unit_exp<T, Exponent>;
};

template <class T>
class canonicalize_unit_product {
 public:
  using type = unit_product<typename canonicalize_unit_exp<T>::type>;
};

template <>
class canonicalize_unit_product<no_unit> {
 public:
  using type = unit_product<>;
};

template <>
class canonicalize_unit_product<unit_product<>> {
 public:
  using type = unit_product<>;
};

template <class FirstUnit, class... OtherUnits>
class canonicalize_unit_product<unit_product<FirstUnit, OtherUnits...>> {
 public:
  using type = typename prepend_unit_product<
    typename canonicalize_unit_exp<FirstUnit>::type,
    typename canonicalize_unit_product<unit_product<OtherUnits...>>::type>::type;
};

template <class T>
class simplify_unit_exp;

template <class T, int Exponent>
class simplify_unit_exp<unit_exp<T, Exponent>> {
 public:
  using type = unit_exp<T, Exponent>;
};

template <class T>
class simplify_unit_exp<unit_exp<T, 1>> {
 public:
  using type = T;
};

template <class T>
class simplify_unit_exp<unit_exp<T, 0>> {
 public:
  using type = no_unit;
};

template <int Exponent>
class simplify_unit_exp<unit_exp<no_unit, Exponent>> {
 public:
  using type = no_unit;
};

template <>
class simplify_unit_exp<unit_exp<no_unit, 1>> {
 public:
  using type = no_unit;
};

template <>
class simplify_unit_exp<unit_exp<no_unit, 0>> {
 public:
  using type = no_unit;
};

template <class T>
class simplify_unit_product {
 public:
  using type = T;
};

template <>
class simplify_unit_product<unit_product<>> {
 public:
  using type = no_unit;
};

template <class FirstUnit, class... OtherUnits>
class simplify_unit_product<unit_product<FirstUnit, OtherUnits...>> {
 public:
  using type = typename prepend_unit_product<
    typename simplify_unit_exp<FirstUnit>::type,
    typename simplify_unit_product<unit_product<OtherUnits...>>::type>::type;
};

template <class A, class B>
class multiply_canonical_unit_product_exp;

template <class UnitExp>
class multiply_canonical_unit_product_exp<
  unit_product<>,
  UnitExp>
{
 public:
  using type = unit_product<UnitExp>;
};

template <class Unit, int Exponent1, int Exponent2, class... OtherUnits>
class multiply_canonical_unit_product_exp<
  unit_product<unit_exp<Unit, Exponent1>, OtherUnits...>,
  unit_exp<Unit, Exponent2>>
{
 public:
  using type = typename prepend_unit_product<
    typename simplify_unit_exp<unit_exp<Unit, Exponent1 + Exponent2>>::type,
    unit_product<OtherUnits...>>::type;
};

template <class FirstUnit, class... OtherUnits, class UnitExp>
class multiply_canonical_unit_product_exp<
  unit_product<FirstUnit, OtherUnits...>,
  UnitExp>
{
 public:
  using type = typename prepend_unit_product<
    FirstUnit,
    typename multiply_canonical_unit_product_exp<unit_product<OtherUnits...>, UnitExp>::type>::type;
};

template <class A, class B>
class multiply_canonical_unit_products;

template <class Product>
class multiply_canonical_unit_products<
  Product,
  unit_product<>>
{
 public:
  using type = Product;
};

template <class FirstUnit, class... OtherUnits, class Product>
class multiply_canonical_unit_products<
  Product,
  unit_product<FirstUnit, OtherUnits...>>
{
 public:
  using type = typename multiply_canonical_unit_products<
    typename multiply_canonical_unit_product_exp<Product, FirstUnit>::type,
    unit_product<OtherUnits...>>::type;
};

template <class A>
class invert_canonical_unit_product;

template <>
class invert_canonical_unit_product<unit_product<>>
{
 public:
  using type = unit_product<>;
};

template <class FirstUnit, int Exponent, class... OtherUnits>
class invert_canonical_unit_product<unit_product<unit_exp<FirstUnit, Exponent>, OtherUnits...>>
{
 public:
  using type = typename prepend_unit_product<
    unit_exp<FirstUnit, -Exponent>,
    typename invert_canonical_unit_product<unit_product<OtherUnits...>>::type>::type;
};

template <class A, int Root>
class canonical_unit_product_root;

template <int Root>
class canonical_unit_product_root<unit_product<>, Root>
{
 public:
  using type = unit_product<>;
};

template <class FirstUnit, int Exponent, class... OtherUnits, int Root>
class canonical_unit_product_root<unit_product<unit_exp<FirstUnit, Exponent>, OtherUnits...>, Root>
{
 public:
  static_assert(Exponent % Root == 0, "named unit term not divisible when taking root");
  using type = typename prepend_unit_product<
    unit_exp<FirstUnit, Exponent / Root>,
    typename canonical_unit_product_root<unit_product<OtherUnits...>, Root>::type>::type;
};

template< class, class = void >
struct has_dimension : std::false_type { };
template< class T >
struct has_dimension<T, std::void_t<typename T::dimension>> : std::true_type { };

template< class, class = void >
struct has_magnitude : std::false_type { };
template< class T >
struct has_magnitude<T, std::void_t<typename T::magnitude>> : std::true_type { };

}

namespace details {

inline std::string trailing_positive_unit_product_name(unit_product<>)
{
  return std::string();
}

template <class FirstUnit, int Exponent, class... OtherUnits,
  std::enable_if_t<(Exponent < 0), bool> = false> 
std::string trailing_positive_unit_product_name(unit_product<unit_exp<FirstUnit, Exponent>, OtherUnits...>)
{
  return trailing_positive_unit_product_name(unit_product<OtherUnits...>());
}

template <class FirstUnit, class... OtherUnits> 
std::string trailing_positive_unit_product_name(unit_product<FirstUnit, OtherUnits...>)
{
  return std::string("*") + FirstUnit::name() + trailing_positive_unit_product_name(unit_product<OtherUnits...>());
}

template <class FirstUnit, int Exponent,
  std::enable_if_t<(Exponent < 0), bool> = false> 
std::string positive_unit_product_name(unit_exp<FirstUnit, Exponent>)
{
  return std::string();
}

template <class Unit> 
std::string positive_unit_product_name(Unit)
{
  return Unit::name();
}

template <class FirstUnit, int Exponent, class... OtherUnits,
  std::enable_if_t<(Exponent < 0), bool> = false> 
std::string positive_unit_product_name(unit_product<unit_exp<FirstUnit, Exponent>, OtherUnits...>)
{
  return positive_unit_product_name(unit_product<OtherUnits...>());
}

template <class FirstUnit, class... OtherUnits> 
std::string positive_unit_product_name(unit_product<FirstUnit, OtherUnits...>)
{
  return FirstUnit::name() + trailing_positive_unit_product_name(unit_product<OtherUnits...>());
}

inline std::string trailing_negative_unit_product_name(unit_product<>)
{
  return std::string();
}

template <class FirstUnit, int Exponent, class... OtherUnits,
  std::enable_if_t<(Exponent < 0), bool> = false> 
std::string trailing_negative_unit_product_name(unit_product<unit_exp<FirstUnit, Exponent>, OtherUnits...>)
{
  using simplified_unit = typename simplify_unit_exp<unit_exp<FirstUnit, -Exponent>>::type;
  return std::string("/") + simplified_unit::name() +
    trailing_negative_unit_product_name(unit_product<OtherUnits...>());
}

template <class FirstUnit, class... OtherUnits> 
std::string trailing_negative_unit_product_name(unit_product<FirstUnit, OtherUnits...>)
{
  return trailing_negative_unit_product_name(unit_product<OtherUnits...>());
}

template <class FirstUnit, int Exponent,
  std::enable_if_t<(Exponent < 0), bool> = false> 
std::string trailing_negative_unit_product_name(unit_exp<FirstUnit, Exponent>)
{
  using simplified_unit = typename simplify_unit_exp<unit_exp<FirstUnit, -Exponent>>::type;
  return std::string("/") + simplified_unit::name();
}

template <class FirstUnit> 
std::string trailing_negative_unit_product_name(FirstUnit)
{
  return std::string();
}

template <class... Units>
std::string unit_product_name(unit_product<Units...> product)
{
  auto result = positive_unit_product_name(product);
  if (result.empty()) result = no_unit::name();
  result += trailing_negative_unit_product_name(product);
  return result;
}

}

template <class LastUnit>
std::string unit_product<LastUnit>::name()
{
  return details::unit_product_name(unit_product<LastUnit>());
}

template <class FirstUnit, class... OtherUnits>
std::string unit_product<FirstUnit, OtherUnits...>::name()
{
  return details::unit_product_name(unit_product<FirstUnit, OtherUnits...>());
}

// Section 6: some type traits helpers for identifying and comparing units

template <class A>
inline constexpr bool is_unit =
  details::has_dimension<A>::value &&
  details::has_magnitude<A>::value;

template <class A, class B>
inline constexpr bool is_same_unit =
  std::is_same_v<typename A::dimension, typename B::dimension> &&
  std::is_same_v<typename A::magnitude, typename B::magnitude>;

// Section 7: basic math operations for units: multiply/divide/root

template <class A, class B>
using unit_multiply = typename details::simplify_unit_product<
  typename details::multiply_canonical_unit_products<
    typename details::canonicalize_unit_product<A>::type,
    typename details::canonicalize_unit_product<B>::type>::type>::type;

template <class A, class B>
using unit_divide = typename details::simplify_unit_product<
  typename details::multiply_canonical_unit_products<
    typename details::canonicalize_unit_product<A>::type,
    typename details::invert_canonical_unit_product<
      typename details::canonicalize_unit_product<B>::type>::type>::type>::type;

template <class A, int Root>
using unit_root = typename details::simplify_unit_product<
    typename details::canonical_unit_product_root<
      typename details::canonicalize_unit_product<A>::type,
      Root>::type>::type;

template <class A>
using unit_inverse = typename details::simplify_unit_product<
    typename details::invert_canonical_unit_product<
      typename details::canonicalize_unit_product<A>::type>::type>::type;

// Section 8: derived units based on the named units, prefixes, and math operations

using millisecond = milli<second>;
using microsecond = micro<second>;
using nanosecond = nano<second>;

using centimeter = centi<meter>;
using millimeter = milli<meter>;
using micrometer = micro<meter>;
using nanometer = nano<meter>;

using kilogram = kilo<gram>;

using reciprocal_second = unit_inverse<second>;
using meter_per_second = unit_divide<meter, second>;
using meter_per_second_squared = unit_divide<meter_per_second, second>;

using reciprocal_meter = unit_inverse<meter>;
using square_meter = unit_multiply<meter, meter>;
using square_centimeter = unit_multiply<centimeter, centimeter>;

using cubic_meter = unit_multiply<meter, square_meter>;
using cubic_centimeter = unit_multiply<centimeter, square_centimeter>;

using per_square_meter = unit_inverse<square_meter>;
using per_cubic_meter = unit_inverse<cubic_meter>;

using kilogram_per_cubic_meter = unit_divide<kilogram, cubic_meter>;
using gram_per_cubic_centimeter = unit_divide<gram, cubic_centimeter>;

using megajoule = mega<joule>;

using joule_per_cubic_meter = unit_divide<joule, cubic_meter>;
using joule_per_kilogram = unit_divide<joule, kilogram>;
using megajoule_per_kilogram = unit_divide<megajoule, kilogram>;
using joule_per_kilogram_per_kelvin = unit_divide<joule_per_kilogram, degree_kelvin>;

using gigapascal = giga<pascal>;

using newton_second = unit_multiply<newton, second>;

using square_meter_per_second = unit_divide<square_meter, second>;
using pascal_second = unit_multiply<pascal, second>;

using ohm_meter = unit_multiply<ohm, meter>;
using siemens_per_meter = unit_divide<siemens, meter>;

using gaussian_electrical_conductivity_unit = unit_inverse<gaussian_electrical_resistivity_unit>;

}
