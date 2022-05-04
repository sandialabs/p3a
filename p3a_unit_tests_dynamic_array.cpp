#include "gtest/gtest.h"

#include "p3a_dynamic_array.hpp" 

TEST(dynamic_array, device_fill)
{
  p3a::device_array<double> a;
  a.resize(100, -42.0);
}
