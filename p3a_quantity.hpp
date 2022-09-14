#pragma once

#include "p3a_scalar.hpp"
#include "p3a_constants.hpp"

#include "kul.hpp"

#include "p3a_simd.hpp"

namespace p3a {

// unit types

using kul::reciprocal;

using kul::second;
using kul::meter;
using kul::cubic_meter;
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
using kul::reciprocal_meters;
using kul::square_meters;
using kul::cubic_meters;
using kul::kilograms;
using kul::meters_per_second;
using kul::meters_per_second_squared;
using kul::kelvins;
using kul::amperes;
using kul::pascals;
using kul::gigapascals;
using kul::joules;
using kul::watts;
using kul::newtons;
using kul::kilograms_per_cubic_meter;
using kul::kilogram_meters_per_second;
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
using kul::pascal_seconds;

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
using kul::asin;
using kul::acos;

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

template <class T, class Unit>
struct zero<quantity<T, Unit>> {
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline static constexpr
  quantity<T, Unit> value() { return quantity<T, Unit>(zero_value<T>()); }
};

}

template <class M, class T, class Unit,
         std::enable_if_t<!std::is_same_v<M, bool>, bool> = false>
P3A_HOST_DEVICE P3A_ALWAYS_INLINE
auto condition(M const& mask, quantity<T, Unit> const& a, quantity<T, Unit> const& b)
{
  return quantity<T, Unit>(condition(mask, a.value(), b.value()));
}

template <class T, class Unit, class Abi>
P3A_HOST_DEVICE P3A_ALWAYS_INLINE
auto load(quantity<T, Unit> const* ptr, int offset, simd_mask<T, Abi> const& mask)
{
  return quantity<simd<T, Abi>, Unit>(load(&(ptr->value()), offset, mask));
}

template <class T, class Unit, class Abi>
P3A_HOST_DEVICE P3A_ALWAYS_INLINE
void store(quantity<simd<T, Abi>, Unit> const& q, quantity<T, Unit>* ptr, int offset, simd_mask<T, Abi> const& mask)
{
  store(q.value(), &(ptr->value()), offset, mask);
}

}
