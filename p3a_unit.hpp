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

template <class Dimension, class MagnitudeInSI>
class unit {
 public:
  using dimension = Dimension;
  using magnitude_in_si = MagnitudeInSI; 
};

}
