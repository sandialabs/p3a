#pragma once

#include <ratio>

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
 * the MagnitudeInSI parameter should be std::ratio<1000, 1>
 * because one kilometer equals 1000 meters and the meter
 * is the SI unit of length.
 */

template <class Dimension, class MagnitudeInSI = std::ratio<1>>
class unit {
 public:
  using dimension = Dimension;
  using magnitude_in_si = MagnitudeInSI; 
};

template <class Prefix, class Unit>
using prefix_unit = unit<
    typename Unit::dimension,
    std::ratio_multiply<Prefix, typename Unit::magnitude_in_si>>;

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

template <class Dimension, class MagnitudeInSI>
class nano<unit<Dimension, MagnitudeInSI>> {
 public:
  using type = prefix_unit<std::nano, unit<Dimension, MagnitudeInSI>>;
};

template <class Dimension, class MagnitudeInSI>
class micro<unit<Dimension, MagnitudeInSI>> {
 public:
  using type = prefix_unit<std::micro, unit<Dimension, MagnitudeInSI>>;
};

template <class Dimension, class MagnitudeInSI>
class milli<unit<Dimension, MagnitudeInSI>> {
 public:
  using type = prefix_unit<std::milli, unit<Dimension, MagnitudeInSI>>;
};

template <class Dimension, class MagnitudeInSI>
class centi<unit<Dimension, MagnitudeInSI>> {
 public:
  using type = prefix_unit<std::centi, unit<Dimension, MagnitudeInSI>>;
};

template <class Dimension, class MagnitudeInSI>
class kilo<unit<Dimension, MagnitudeInSI>> {
 public:
  using type = prefix_unit<std::kilo, unit<Dimension, MagnitudeInSI>>;
};

template <class Dimension, class MagnitudeInSI>
class mega<unit<Dimension, MagnitudeInSI>> {
 public:
  using type = prefix_unit<std::mega, unit<Dimension, MagnitudeInSI>>;
};

template <class Dimension, class MagnitudeInSI>
class giga<unit<Dimension, MagnitudeInSI>> {
 public:
  using type = prefix_unit<std::giga, unit<Dimension, MagnitudeInSI>>;
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

// some common SI units

using second = unit<p3a::time>;
using millisecond = milli<second>;
using microsecond = micro<second>;
using nanosecond = nano<second>;

using meter = unit<length>;
using centimeter = centi<meter>;
using millimeter = milli<meter>;
using micrometer = micro<meter>;
using nanometer = nano<meter>;

using gram = unit<mass, std::ratio<1, 1000>>;
using kilogram = kilo<gram>;

using ampere = unit<electric_current>;

using kelvin = unit<temperature>;

using mole = unit<amount_of_substance>;

using candela = unit<luminous_intensity>;

// variations of dimensionless quantities

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

}
