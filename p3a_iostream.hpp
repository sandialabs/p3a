#pragma once

#include "p3a_quantity.hpp"

#include <iosfwd>

namespace p3a {

template <class Unit, std::enable_if_t<is_unit<Unit>, bool> = false>
std::ostream& operator<<(std::ostream& s, Unit)
{
  s << Unit::name();
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
