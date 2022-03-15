#pragma once

#include "p3a_macros.hpp"

namespace p3a {

/* This class represents, as a C++ object (not just a type),
 * a physical quantity.
 *
 * The first template argument, Unit, should be an instance of
 * the class template p3a::unit describing the unit of measurement
 * in which the quantity is represented.
 *
 * The second template argument, Representation, should be the type
 * used to store the numerical representation of the quantity
 * as a factor to multiply by the unit of measure.
 * Representation is most often a floating-point type of some sort.
 *
 * The third template argument handles the subtle distinction between
 * relative and absolute quantities, for example the difference between
 * a point in space versus a vector in space, or the difference between
 * an absolute temperature versus a change in temperature.
 * Mathematically, quantities can be thought of as forming an affine space
 * with points and translation vectors.
 * In P3A, if OriginInSI is void, then this quantity is a "vector quantity",
 * which means a relative quantity such as a displacement.
 * If OriginInSI is an instance of std::ratio, then this quantity is a
 * "point quantity", which means an absolute quantity such as a position.
 * The actual rational number represented by the std::ratio expresses a
 * compile-time origin, measured in the SI unit used to measure quantities
 * of this dimension.
 * This is useful, for example, to describe how Kelvin, Celcius, and
 * Fahrenheit scales have different definitions of zero.
 */

template <class Unit, class Representation = double, class OriginInSI = void>
class quantity {
 public:
  using value_type = Representation;
  using unit = Unit;
 private:
  value_type m_value;
 public:
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  explicit quantity(value_type const& r)
    :m_value(r)
  {}
  P3A_ALWAYS_INLINE inline
  quantity() = default;
  P3A_ALWAYS_INLINE inline constexpr
  quantity(quantity const&) = default;
  P3A_ALWAYS_INLINE inline constexpr
  quantity(quantity&&) = default;
  P3A_ALWAYS_INLINE inline constexpr
  quantity& operator=(quantity const&) = default;
  P3A_ALWAYS_INLINE inline constexpr
  quantity& operator=(quantity&&) = default;
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  value_type const& value() const { return m_value; }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  value_type& value() { return m_value; }
};

}
