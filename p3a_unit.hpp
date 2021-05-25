#pragma once

#include <ratio>

namespace p3a {

template <class T>
class units {
  // the measurement system's base units
  // expressed in terms of SI units
  T m_time_unit;
  T m_length_unit;
  T m_mass_unit;
  T m_current_unit;
  T m_temperature_unit;
  T m_amount_unit;
  T m_intensity_unit;
 public:
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
  explicit units(
      T const& time_unit_arg = T(1),
      T const& length_unit_arg = T(1),
      T const& mass_unit_arg = T(1),
      T const& current_unit_arg = T(1),
      T const& temperature_unit_arg = T(1),
      T const& amount_unit_arg = T(1),
      T const& intensity_unit_arg = T(1))
    :m_time_unit(time_unit_arg)
    ,m_length_unit(length_unit_arg)
    ,m_mass_unit(mass_unit_arg)
    ,m_current_unit(current_unit_arg)
    ,m_temperature_unit(temperature_unit_arg)
    ,m_amount_unit(amount_unit_arg)
    ,m_intensity_unit(intensity_unit_arg)
  {
  }
  template <class Dimension>
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
  T unit() const
  {
    T result(1);
    for (int i = 0; i < Dimension::time_power; ++i) {
      result *= m_time_unit;
    }
    for (int i = 0; i < -Dimension::time_power; ++i) {
      result /= m_time_unit;
    }
    for (int i = 0; i < Dimension::length_power; ++i) {
      result *= m_length_unit;
    }
    for (int i = 0; i < -Dimension::length_power; ++i) {
      result /= m_length_unit;
    }
    for (int i = 0; i < Dimension::mass_power; ++i) {
      result *= m_mass_unit;
    }
    for (int i = 0; i < -Dimension::mass_power; ++i) {
      result /= m_mass_unit;
    }
    for (int i = 0; i < Dimension::current_power; ++i) {
      result *= m_current_unit;
    }
    for (int i = 0; i < -Dimension::current_power; ++i) {
      result /= m_current_unit;
    }
    for (int i = 0; i < Dimension::temperature_power; ++i) {
      result *= m_temperature_unit;
    }
    for (int i = 0; i < -Dimension::temperature_power; ++i) {
      result /= m_temperature_unit;
    }
    for (int i = 0; i < Dimension::amount_power; ++i) {
      result *= m_amount_unit;
    }
    for (int i = 0; i < -Dimension::amount_power; ++i) {
      result /= m_amount_unit;
    }
    for (int i = 0; i < Dimension::intensity_power; ++i) {
      result *= m_intensity_unit;
    }
    for (int i = 0; i < -Dimension::intensity_power; ++i) {
      result /= m_intensity_unit;
    }
    return result;
  }
};

template <class T>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
units<T> centimetre_gram_second_units()
{
  return units<T>(
      T(1),
      T(1) / T(100),
      T(1) / T(1000));
}

template <class T>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
units<T> electrostatic_units()
{
  return units<T>(
      T(1),
      T(1) / T(100),
      T(1) / T(1000),
// one (statcoulomb or Franklin) is 10/c Amperes,
// where c is the speed of light in CGS units,
// so 1/(10 * c) when c is expressed in SI units
      T(1) / (T(10) * speed_of_light_value<T>()));
}

}
