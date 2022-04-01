#pragma once

namespace p3a {

/* This class represents, as a C++ type, a physical dimension as defined here:
 * https://en.wikipedia.org/wiki/Dimensional_analysis#Mathematical_formulation
 *
 * A a set of seven integral exponents, one for each of the seven base
 * units of the International System of Units (SI)
 */

template <
  int TimeExponent = 0,
  int LengthExponent = 0,
  int MassExponent = 0,
  int ElectricCurrentExponent = 0,
  int TemperatureExponent = 0,
  int AmountOfSubstanceExponent = 0,
  int LuminousIntensityExponent = 0>
class dimension {
 public:
  inline static constexpr int time_exponent = TimeExponent;
  inline static constexpr int length_exponent = LengthExponent;
  inline static constexpr int mass_exponent = MassExponent;
  inline static constexpr int electric_current_exponent = ElectricCurrentExponent;
  inline static constexpr int temperature_exponent = TemperatureExponent;
  inline static constexpr int amount_of_substance_exponent = AmountOfSubstanceExponent;
  inline static constexpr int luminous_intensity_exponent = LuminousIntensityExponent;
};

template <class A, class B>
using dimension_multiply = dimension<
  A::time_exponent + B::time_exponent,
  A::length_exponent + B::length_exponent,
  A::mass_exponent + B::mass_exponent,
  A::electric_current_exponent + B::electric_current_exponent,
  A::temperature_exponent + B::temperature_exponent,
  A::amount_of_substance_exponent + B::amount_of_substance_exponent,
  A::luminous_intensity_exponent + B::luminous_intensity_exponent>;

template <class A, class B>
using dimension_divide = dimension<
  A::time_exponent - B::time_exponent,
  A::length_exponent - B::length_exponent,
  A::mass_exponent - B::mass_exponent,
  A::electric_current_exponent - B::electric_current_exponent,
  A::temperature_exponent - B::temperature_exponent,
  A::amount_of_substance_exponent - B::amount_of_substance_exponent,
  A::luminous_intensity_exponent - B::luminous_intensity_exponent>;

namespace details {

template <class Dimension, int Root>
class dimension_root_helper {
 public:
  static_assert(Dimension::time_exponent % Root == 0,
      "time dimension exponent not divisible by root");
  static_assert(Dimension::length_exponent % Root == 0,
      "length dimension exponent not divisible by root");
  static_assert(Dimension::mass_exponent % Root == 0,
      "mass dimension exponent not divisible by root");
  static_assert(Dimension::electric_current_exponent % Root == 0,
      "electric current dimension exponent not divisible by root");
  static_assert(Dimension::temperature_exponent % Root == 0,
      "thermodynamic temperature dimension exponent not divisible by root");
  static_assert(Dimension::amount_of_substance_exponent % Root == 0,
      "amount of substance dimension exponent not divisible by root");
  static_assert(Dimension::luminous_intensity_exponent % Root == 0,
      "luminous intensity dimension exponent not divisible by root");
  using type = dimension<
    Dimension::time_exponent / Root,
    Dimension::length_exponent / Root,
    Dimension::mass_exponent / Root,
    Dimension::electric_current_exponent / Root,
    Dimension::temperature_exponent / Root,
    Dimension::amount_of_substance_exponent / Root,
    Dimension::luminous_intensity_exponent / Root>;
};

}

template <class Dimension, int Root>
using dimension_root = typename details::dimension_root_helper<Dimension, Root>::type;

// the dimension of a dimensionless quantity
using no_dimension = dimension<>;

// base dimensions
using time = dimension<1>;
using length = dimension<0, 1>;
using mass = dimension<0, 0, 1>;
using electric_current = dimension<0, 0, 0, 1>;
using temperature = dimension<0, 0, 0, 0, 1>;
using amount_of_substance = dimension<0, 0, 0, 0, 0, 1>;
using luminous_intensity = dimension<0, 0, 0, 0, 0, 0, 1>;

using frequency = dimension<-1>;
// same dimension as velocity
using speed = dimension<-1, 1, 0>;
using acceleration = dimension<-2, 1, 0>;

using area = dimension<0, 2>;
using volume = dimension<0, 3>;

// per unit area
using area_density = dimension<0, -2, 0>;
// per unit volume
using volumetric_density = dimension<0, -3, 0>;

// mass per unit volume
using volumetric_mass_density = dimension<0, -3, 1>;

using energy = dimension<-2, 2, 1>;
// energy per unit time
using power = dimension<-3, 2, 1>;
// energy per unit volume
using volumetric_energy_density = dimension<-2, -1, 1>;
// energy per unit mass
using specific_energy = dimension<-2, 2, 0>;
// energy per unit mass per unit time
using specific_energy_rate = dimension<-3, 2, 0>;

using force = dimension<-2, 1, 1>;
// force per unit area
using pressure = dimension<-2, -1, 1>;

using momentum = dimension<-1, 1, 1>;
// momentum per unit volume
using volumetric_momentum_density = dimension<-1, -2, 1>;

using kinematic_viscosity = dimension<-1, 2, 0>;
using dynamic_viscosity = dimension<-1, -1, 1>;

using specific_heat = dimension<-2, 2, 0, 0, -1>;

// electromagnetic dimensions

using electric_charge = dimension<1, 0, 0, 1>;
using electric_potential = dimension<-3, 2, 1, -1>;
using electrical_resistance = dimension<-3, 2, 1, -2>;
using electrical_conductance = dimension<3, -2, -1, 2>;
using electrical_resistivity = dimension<-3, 3, 1, -2>;
using electrical_conductivity = dimension<3, -3, -1, 2>;
using inductance = dimension<-2, 2, 1, -2>;
using capacitance = dimension<4, -2, -1, 2>;

}
