#include "gtest/gtest.h"

#include <string>

#include "p3a_dynamic_array.hpp" 

TEST(dynamic_array, device_fill)
{
  p3a::device_array<double> a;
  a.resize(100, -42.0);
}

TEST(dynamic_array, fancy_object)
{
  p3a::dynamic_array<std::string> a;
  a.push_back("a");
  a.push_back("b");
}
