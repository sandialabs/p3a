#include "gtest/gtest.h"

TEST(my_test_suite, two_plus_two)
{
  int sum = 2 + 2;
  EXPECT_EQ(sum, 4);
}
