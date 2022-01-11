#include "gtest/gtest.h"
#include "p3a_functions.hpp"

using namespace p3a;

TEST(fp64, exponent){
  double const input = 1.25000000000000000e-01;
  int const ours = p3a::exponent(input);
  int theirs;
  std::frexp(input, &theirs);
  EXPECT_EQ(ours, theirs - 1);
  double const pot = p3a::power_of_two_as_double(ours);
  EXPECT_EQ(pot, input);
}

