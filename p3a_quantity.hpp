#pragma once

#include "kul.hpp"

namespace p3a {

using kul::kilogram;
using kul::kelvin;
using kul::pascal;
using kul::joule;
using kul::kilogram_per_cubic_meter;
using kul::joule_per_kilogram;

using kul::quantity;

using kul::kilograms;
using kul::meters_per_second;
using kul::kelvins;
using kul::pascals;
using kul::joules;
using kul::kilograms_per_cubic_meter;
using kul::joules_per_kilogram;
using kul::siemens_per_meter_quantity;

template <class A, class B>
using unit_divide = kul::divide<A, B>;

}
