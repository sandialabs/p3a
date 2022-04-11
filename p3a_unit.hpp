#pragma once

#include <ratio>

#include "p3a_dimension.hpp"

namespace p3a {

/* This class represents, as a C++ type, a unit of measurement.
 *
 * It is a physical dimension, represented by an instance of
 * the p3a::dimension class template, and a magnitude.
 * The magnitude should be an instance of the std::ratio class
 * template, which represents a rational number.
 * That rational number should be the magnitude of the unit
 * being described, as measured in the corresponding SI unit.
 * For example, if describing the kilometer unit of length,
 * the Magnitude parameter should be std::ratio<1000, 1>
 * because one kilometer equals 1000 meters and the meter
 * is the SI unit of length.
 */

template <class Dimension, class Magnitude = std::ratio<1>>
class unit {
 public:
  using dimension = Dimension;
  using magnitude = Magnitude; 
};

template <class A, class B>
using unit_multiply = unit<
    dimension_multiply<typename A::dimension, typename B::dimension>,
    std::ratio_multiply<typename A::magnitude, typename B::magnitude>>;

template <class A, class B>
using unit_divide = unit<
    dimension_divide<typename A::dimension, typename B::dimension>,
    std::ratio_divide<typename A::magnitude, typename B::magnitude>>;

namespace details {

template <class Ratio, int Root>
class ratio_root {
  static_assert(std::is_same_v<typename Ratio::type, std::ratio<1>>,
      "taking roots of std::ratio other than one is not supported yet"); 
 public:
  using type = std::ratio<1>;
};

}

template <class Ratio, int Root>
using ratio_root = typename details::ratio_root<Ratio, Root>::type;

template <class A, int Root>
using unit_root = unit<
    dimension_root<typename A::dimension, Root>,
    ratio_root<typename A::magnitude, Root>>;

template <class Ratio, class Unit>
using unit_scale = unit<
    typename Unit::dimension,
    std::ratio_multiply<Ratio, typename Unit::magnitude>>;

namespace details {

// we use class template specialization for the metric prefixes
// so that we can do both mega<unit<...>> and mega<quantity<...>>

template <class T> class nano;
template <class T> class micro;
template <class T> class milli;
template <class T> class centi;
template <class T> class kilo;
template <class T> class mega;
template <class T> class giga;

template <class Dimension, class Magnitude>
class nano<unit<Dimension, Magnitude>> {
 public:
  using type = unit_scale<std::nano, unit<Dimension, Magnitude>>;
};

template <class Dimension, class Magnitude>
class micro<unit<Dimension, Magnitude>> {
 public:
  using type = unit_scale<std::micro, unit<Dimension, Magnitude>>;
};

template <class Dimension, class Magnitude>
class milli<unit<Dimension, Magnitude>> {
 public:
  using type = unit_scale<std::milli, unit<Dimension, Magnitude>>;
};

template <class Dimension, class Magnitude>
class centi<unit<Dimension, Magnitude>> {
 public:
  using type = unit_scale<std::centi, unit<Dimension, Magnitude>>;
};

template <class Dimension, class Magnitude>
class kilo<unit<Dimension, Magnitude>> {
 public:
  using type = unit_scale<std::kilo, unit<Dimension, Magnitude>>;
};

template <class Dimension, class Magnitude>
class mega<unit<Dimension, Magnitude>> {
 public:
  using type = unit_scale<std::mega, unit<Dimension, Magnitude>>;
};

template <class Dimension, class Magnitude>
class giga<unit<Dimension, Magnitude>> {
 public:
  using type = unit_scale<std::giga, unit<Dimension, Magnitude>>;
};

}

// a few common metric prefixes

template <class T>
using nano = typename details::nano<T>::type;
template <class T>
using micro = typename details::micro<T>::type;
template <class T>
using milli = typename details::milli<T>::type;
template <class T>
using centi = typename details::centi<T>::type;
template <class T>
using kilo = typename details::kilo<T>::type;
template <class T>
using mega = typename details::mega<T>::type;
template <class T>
using giga = typename details::giga<T>::type;

// the unit of a unitless quantity
using no_unit = unit<no_dimension>;

template <class A>
using unit_inverse = unit_divide<no_unit, A>;

using second = unit<p3a::time>;
using millisecond = milli<second>;
using microsecond = micro<second>;
using nanosecond = nano<second>;

using meter = unit<length>;
using centimeter = centi<meter>;
using millimeter = milli<meter>;
using micrometer = micro<meter>;
using nanometer = nano<meter>;

using inch = unit<length,
      std::ratio_multiply<std::ratio<254, 10>, millimeter::magnitude>>;

using gram = unit<mass, std::ratio<1, 1000>>;
using kilogram = kilo<gram>;

using ampere = unit<electric_current>;

using degree_kelvin = unit<temperature>;
using degree_celcius = unit<temperature>;
using degree_fahrenheit = unit<temperature, std::ratio<5, 9>>;
using temperature_electronvolt = unit<temperature,
      std::ratio<1160451812, 100000>>;

using mole = unit<amount_of_substance>;

using candela = unit<luminous_intensity>;

using reciprocal_second = unit<reciprocal_time>;
using hertz = unit<frequency>;
using meter_per_second = unit<speed>;
using meter_per_second_squared = unit<acceleration>;

using reciprocal_meter = unit<reciprocal_length>;
using square_meter = unit<area>;
using square_centimeter = unit_multiply<centimeter, centimeter>;

using cubic_meter = unit<volume>;
using cubic_centimeter = unit_multiply<centimeter, square_centimeter>;

using per_square_meter = unit<area_density>;
using per_cubic_meter = unit<volumetric_density>;

using kilogram_per_cubic_meter = unit<volumetric_mass_density>;
using gram_per_cubic_centimeter = unit_divide<gram, cubic_centimeter>;

using joule = unit<energy>;
using megajoule = mega<joule>;

using watt = unit<power>;

using joule_per_cubic_meter = unit<volumetric_energy_density>;
using joule_per_kilogram = unit<specific_energy>;
using megajoule_per_kilogram = unit_divide<megajoule, kilogram>;
using joule_per_kilogram_per_kelvin = unit<specific_heat>;

using newton = unit<force>;

using pascal = unit<pressure>;
using gigapascal = giga<pascal>;

using newton_second = unit<momentum>;

using square_meter_per_second = unit<kinematic_viscosity>;
using pascal_second = unit<dynamic_viscosity>;

using radian = no_unit;

namespace details {

using pi_ratio = std::ratio<
  31415926535897932,
  10000000000000000>;

}

// the degree unit for measuring circular arc has no dimension,
// but it does have a magnitude that is not one: it is pi over 180
using arc_degree = unit<no_dimension,
      std::ratio_divide<details::pi_ratio, std::ratio<180>>>;

using percent = unit<no_dimension, std::ratio<1, 100>>;

using volt = unit<electric_potential>;
using siemens = unit<electrical_conductance>;
using siemens_per_meter = unit<electrical_conductivity>;
using ohm = unit<electrical_resistance>;
using ohm_meter = unit<electrical_resistivity>;
using henry = unit<inductance>;
using farad = unit<capacitance>;

namespace details {
  using speed_of_light_in_centimeters_per_second = std::ratio<29979245800>;
}

// The so-called "second" of electrical resistivity used in Gaussian units
// is equivalent to c^2 / 10^11 Ohm-meters
// where c is the speed of light in centimeters per second
// https://en.wikipedia.org/wiki/Centimetre%E2%80%93gram%E2%80%93second_system_of_units#Electromagnetic_units_in_various_CGS_systems
using gaussian_electrical_resistivity_unit =
  unit<electrical_resistivity,
    std::ratio_multiply<
      details::speed_of_light_in_centimeters_per_second,
      std::ratio_divide<
        details::speed_of_light_in_centimeters_per_second,
        std::ratio<100000000000>>>>;

using gaussian_electrical_conductivity_unit =
  unit_divide<no_unit, gaussian_electrical_resistivity_unit>;

}
