#pragma once

#include "p3a_quantity.hpp"

#include <iostream>

#warning "this did get included!"

template <class T, class Unit>
std::ostream& operator<<(std::ostream& s, p3a::quantity<T, Unit> const& q)
{
  s << q.value() << " " << q.unit_name();
  return s;
}
