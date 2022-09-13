#pragma once

#include "p3a_quantity.hpp"

#include <iosfwd>

namespace p3a {

template <class Unit, class ValueType>
std::ostream& operator<<(std::ostream& s, quantity<ValueType, Unit> const& q)
{
  s << q.value() << " " << q.unit_name();
  return s;
}

}
