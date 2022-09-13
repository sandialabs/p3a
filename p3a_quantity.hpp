#pragma once

#include "p3a_scalar.hpp"
#include "p3a_constants.hpp"

#include "kul.hpp"

namespace p3a {

// unit types

using kul::reciprocal;

using kul::second;
using kul::kilogram;
using kul::kelvin;
using kul::pascal;
using kul::gigapascal;
using kul::joule;
using kul::kilogram_per_cubic_meter;
using kul::joule_per_kilogram;
using kul::joule_per_kilogram_per_kelvin;
using kul::siemens_per_meter;

using kul::quantity;
using kul::unitless;

// quantity alias templates

using kul::seconds;
using kul::reciprocal_seconds;
using kul::meters;
using kul::kilograms;
using kul::meters_per_second;
using kul::kelvins;
using kul::amperes;
using kul::pascals;
using kul::gigapascals;
using kul::joules;
using kul::kilograms_per_cubic_meter;
using kul::grams_per_cubic_centimeter;
using kul::joules_per_kilogram;
using kul::joules_per_kilogram_per_kelvin;
using kul::megajoules_per_kilogram;
using kul::volts;
using kul::ohms;
using kul::siemens_quantity;
using kul::farads;
using kul::henries;
using kul::siemens_per_meter_quantity;
using kul::temperature_electronvolts;
using kul::gaussian_conductivity;

template <class A, class B>
using unit_divide = kul::divide<A, B>;

namespace quantity_literals = kul::literals;

using kul::abs;
using kul::pow;
using kul::exp;
using kul::sqrt;
using kul::cbrt;
using kul::sin;
using kul::cos;

namespace details {

template <class T, class Unit>
struct is_scalar<quantity<T, Unit>> {
  inline static constexpr bool value = true;
};

}

namespace constants {

template <class T, class Unit>
struct epsilon<quantity<T, Unit>> {
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline static constexpr
  quantity<T, Unit> value() { return quantity<T, Unit>(epsilon_value<T>()); }
};

}

}
