#pragma once

#include "p3a_quantity.hpp"

#include <iosfwd>

namespace p3a {

namespace details {

inline void format_positive_base_unit(std::ostream& s, bool& printed_one, int exponent, char const* name)
{
  if (exponent == 0) return;
  if (exponent < 0) return;
  if (printed_one) {
    s << " ";
  } else {
    printed_one = true;
  }
  s << name;
  if (exponent > 1)
  s << "^" << exponent;
}

inline void format_negative_base_unit(std::ostream& s, bool& printed_one, int exponent, char const* name)
{
  if (exponent == 0) return;
  if (exponent > 0) return;
  if (printed_one) {
    s << " ";
  } else {
    s << " / ";
    printed_one = true;
  }
  s << name;
  if (exponent < -1)
  s << "^" << -exponent;
}

}

template <
  int TimeExponent,
  int LengthExponent,
  int MassExponent,
  int ElectricCurrentExponent,
  int TemperatureExponent,
  int AmountOfSubstanceExponent,
  int LuminousIntensityExponent>
std::ostream& operator<<(
    std::ostream& s,
    dimension<
      TimeExponent,
      LengthExponent,
      MassExponent,
      ElectricCurrentExponent,
      TemperatureExponent,
      AmountOfSubstanceExponent,
      LuminousIntensityExponent>)
{
  bool printed_one = false;
  details::format_positive_base_unit(s, printed_one, TimeExponent, "s");
  details::format_positive_base_unit(s, printed_one, LengthExponent, "m");
  details::format_positive_base_unit(s, printed_one, MassExponent, "kg");
  details::format_positive_base_unit(s, printed_one, ElectricCurrentExponent, "A");
  details::format_positive_base_unit(s, printed_one, AmountOfSubstanceExponent, "mol");
  details::format_positive_base_unit(s, printed_one, LuminousIntensityExponent, "cd");
  if (!printed_one) s << "1";
  printed_one = false;
  details::format_negative_base_unit(s, printed_one, TimeExponent, "s");
  details::format_negative_base_unit(s, printed_one, LengthExponent, "m");
  details::format_negative_base_unit(s, printed_one, MassExponent, "kg");
  details::format_negative_base_unit(s, printed_one, ElectricCurrentExponent, "A");
  details::format_negative_base_unit(s, printed_one, AmountOfSubstanceExponent, "mol");
  details::format_negative_base_unit(s, printed_one, LuminousIntensityExponent, "cd");
  return s;
}

template <class Dimension, class Magnitude>
std::ostream& operator<<(std::ostream& s, unit<Dimension, Magnitude>)
{
  if ((Magnitude::num != 1) || (Magnitude::den != 1)) {
    s << " * ";
    s << Magnitude::num;
    if (Magnitude::den != 1) {
      s << "/" << Magnitude::den;
    }
    s << " ";
  }
  s << Dimension();
  return s;
}

template <class Unit, class ValueType, class Origin>
std::ostream& operator<<(std::ostream& s, quantity<Unit, ValueType, Origin> const& q)
{
  s << q.value() << " ";
  s << Unit();
  if constexpr (!std::is_same_v<Origin, void>) {
    s << " + ";
    s << Origin::num;
    if (Origin::den != 1) {
      s << "/" << Origin::den;
    }
    s << Unit();
  }
  return s;
}

}
