#pragma once

#include <ratio>

namespace p3a {

using unit_ratio = std::ratio<1, 1>;
using speed_of_light_ratio = std::ratio<299792458, 1>;

template <
  class TimeRatio,
  class LengthRatio,
  class MassRatio,
  class CurrentRatio = unit_ratio,
  class TemperatureRatio = unit_ratio,
  class AmountRatio = unit_ratio,
  class IntensityRatio = unit_ratio,
class units {
 public:
  using time_ratio = TimeRatio;
  using length_ratio = LengthRatio;
  using mass_ratio = MassRatio;
  using current_ratio = CurrentRatio;
  using temperature_ratio = TemperatureRatio;
  using amount_ratio = AmountRatio;
  using intensity_ratio = IntensityRatio;
};

using si_units = units<
    unit_ratio,
    unit_ratio,
    unit_ratio>;

template <class A, class B>
using ratio_product = typename std::ratio<
    A::num * B::num,
    A::den * B::den>::type;

template <class A, class B>
using ratio_quotient = typename std::ratio<
    A::num * B::den,
    A::den * B::num>::type;

template <class A>
using ratio_inverse = std::ratio<A::den, A::num>;

using cgs_units = units<
    std::centi,
    std::milli,
    unit_ratio,
// one (statcoulomb or Franklin) is 10/c Amperes,
// where c is the speed of light in CGS units,
// so 1/(10 * c) when c is expressed in SI units
    ratio_inverse<ratio_product<std::deca, speed_of_light_ratio>>,
    unit_ratio,
    unit_ratio,
    unit_ratio>

template <class Ratio, int Power>
class nonnegative_ratio_power_helper {
  using one_less = typename nonnegative_ratio_power_helper<Ratio, Power - 1>::type;
 public:
  using type = ratio_product<Ratio, one_less>;
};

template <class Ratio>
class nonnegative_ratio_power_helper<Ratio, 0> {
 public:
  using type = unit_ratio;
};

template <class Ratio, int Power>
using nonnegative_ratio_power = typename nonnegative_ratio_power_helper<Ratio, Power>::type;

template <class Ratio, int Power>
using ratio_power =
  ratio_quotient<
    nonnegative_ratio_power<Ratio, std::max(Power, int(0))>
    nonnegative_ratio_power<Ratio, std::max(-Power, int(0))>>;

template <class FromUnits, class ToUnits>
using conversion_units = units<
  ratio_quotient<FromUnits::time_ratio, ToUnits::time_ratio>,
  ratio_quotient<FromUnits::length_ratio, ToUnits::length_ratio>,
  ratio_quotient<FromUnits::mass_ratio, ToUnits::mass_ratio>,
  ratio_quotient<FromUnits::current_ratio, ToUnits::current_ratio>,
  ratio_quotient<FromUnits::temperature_ratio, ToUnits::temperature_ratio>,
  ratio_quotient<FromUnits::amount_ratio, ToUnits::amount_ratio>,
  ratio_quotient<FromUnits::intensity_ratio, ToUnits::intensity_ratio>>;

template <class Dimension, class ConversionUnits>
using conversion_ratio_from_conversion_units =
  ratio_product<ratio_power<ConversionUnits::time_ratio, Dimension::time_power>,
  ratio_product<ratio_power<ConversionUnits::length_ratio, Dimension::length_power>,
  ratio_product<ratio_power<ConversionUnits::mass_ratio, Dimension::mass_power>,
  ratio_product<ratio_power<ConversionUnits::current_ratio, Dimension::current_power>,
  ratio_product<ratio_power<ConversionUnits::temperature_ratio, Dimension::temperature_power>,
  ratio_product<ratio_power<ConversionUnits::amount_ratio, Dimension::amount_power>,
                ratio_power<ConversionUnits::intensity_ratio, Dimension::intensity_power>>>>>>>;

template <class Dimension, class FromUnits, class ToUnits>
using conversion_ratio = conversion_ratio_from_conversion_units<
  Dimension, conversion_units<FromUnits, ToUnits>>;

template <class ToType, class Ratio> 
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE constexpr
ToType ratio_value() { return ToType(Ratio::num) / ToType(Ratio::den); }
:q

}
