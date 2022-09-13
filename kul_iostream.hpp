#pragma once

#include "kul.hpp"

#include <iostream>

namespace kul {

template <class T, class Unit>
std::ostream& operator<<(std::ostream& s, quantity<T, Unit> const& q)
{
  s << q.value() << " " << q.unit_name();
  return s;
}

}
