#include <gtest/gtest.h>

#include "p3a_dynamic_array.hpp"

TEST(dynamic_array, basic)
{
  p3a::dynamic_array<double> a;
  a.resize(100, 1.0);
}

TEST(dynamic_array, of_strings)
{
  p3a::dynamic_array<std::string> a;
  a.push_back("four");
  a.push_back("five");
}

TEST(dynamic_array, nested)
{
  using type = p3a::dynamic_array<
                  p3a::dynamic_array<
                    p3a::dynamic_array<
                      int>>>;
  type a;
  a.resize(2);
  a[0].resize(2);
  a[0][0].resize(2);
  a[0][1].resize(2);
  a[1].resize(2);
  a[1][0].resize(2);
  a[1][1].resize(2);
  type b;
  b = a;
}
