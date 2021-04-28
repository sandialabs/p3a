#pragma once

namespace p3a {

template <
  int TimePower,
  int LengthPower,
  int MassPower,
  int CurrentPower = 0,
  int TemperaturePower = 0,
  int AmountPower = 0,
  int IntensityPower = 0>
class dimension {
 public:
  inline static constexpr int time_power = TimePower;
  inline static constexpr int length_power = LengthPower;
  inline static constexpr int mass_power = MassPower;
  inline static constexpr int current_power = CurrentPower;
  inline static constexpr int temperature_power = TemperaturePower;
  inline static constexpr int amount_power = AmountPower;
  inline static constexpr int intensity_power = IntensityPower;
};

template <class A, class B>
using dimension_product = dimension<
  A::time_power + B::time_power,
  A::length_power + B::length_power,
  A::mass_power + B::mass_power,
  A::current_power + B::current_power,
  A::temperature_power + B::temperature_power,
  A::amount_power + B::amount_power,
  A::intensity_power + B::intensity_power>;

template <class A, class B>
using dimension_quotient = dimension<
  A::time_power - B::time_power,
  A::length_power - B::length_power,
  A::mass_power - B::mass_power,
  A::current_power - B::current_power,
  A::temperature_power - B::temperature_power,
  A::amount_power - B::amount_power,
  A::intensity_power - B::intensity_power>;

template <class Dimension, int Root>
class dimension_root_helper {
 public:
  static_assert(Dimension::time_power % Root == 0, "invalid time dimension root");
  static_assert(Dimension::length_power % Root == 0, "invalid length dimension root");
  static_assert(Dimension::mass_power % Root == 0, "invalid mass dimension root");
  static_assert(Dimension::current_power % Root == 0, "invalid current dimension root");
  static_assert(Dimension::temperature_power % Root == 0, "invalid temperature dimension root");
  static_assert(Dimension::amount_power % Root == 0, "invalid amount dimension root");
  static_assert(Dimension::intensity_power % Root == 0, "invalid intensity dimension root");
  using type = dimension<
    Dimension::time_power / Root,
    Dimension::length_power / Root,
    Dimension::mass_power / Root,
    Dimension::current_power / Root,
    Dimension::temperature_power / Root,
    Dimension::amount_power / Root,
    Dimension::intensity_power / Root>;
};

template <class Dimension, int Root>
using dimension_root = typename dimension_root_helper<Dimension, Root>::type;

using adimensional = dimension<0, 0, 0>;
using time_dimension = dimension<1, 0, 0>;
using length_dimension = dimension<0, 1, 0>;
using mass_dimension = dimension<0, 0, 1>;
using temperature_dimension = dimension<0, 0, 0, 0, 1>;
using area_dimension = dimension<0, 2, 0>;
using volume_dimension = dimension<0, 3, 0>;
using density_dimension = dimension<0, -3, 0>;
using mass_density_dimension = dimension<0, -3, 1>;
using pressure_dimension = dimension<-2, -1, 1>;
using energy_dimension = dimension<-2, 2, 1>;
using specific_energy_dimension = dimension<-2, 2, 0>;
using specific_energy_rate_dimension = dimension<-3, 2, 0>;
using energy_density_dimension = dimension<-2, -1, 1>;
using velocity_dimension = dimension<-1, 1, 0>;
using momentum_dimension = dimension<-1, 1, 1>;
using momentum_density_dimension = dimension<-1, -2, 1>;
using acceleration_dimension = dimension<-2, 1, 0>;
using force_dimension = dimension<-2, 1, 1>;
using gradient_dimension = dimension<0, -1, 0>;
using rate_dimension = dimension<-1, 0, 0>;
using kinematic_viscosity_dimension = dimension<-1, 2, 0>;
using dynamic_viscosity_dimension = dimension<-1, -1, 1>;
using specific_heat_dimension = dimension<-2, 2, 0, 0, -1>;
using electric_current_dimension = dimension<0, 0, 0, 1>;
using electric_charge_dimension = dimension<1, 0, 0, 1>;
using electric_potential_dimension = dimension<-3, 2, 1, -1>;
using electrical_resistance_dimension = dimension<-3, 2, 1, -2>;
using electrical_conductance_dimension = dimension<3, -2, -1, 2>;
using electrical_resistivity_dimension = dimension<-3, 3, 1, -2>;
using electrical_conductivity_dimension = dimension<3, -3, -1, 2>;

}
