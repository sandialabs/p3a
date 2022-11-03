#include <gtest/gtest.h>
#include <array>

#include "p3a_reduce.hpp"

TEST(reduce, seq)
{
  std::array<int, 5> test_data = {1, 0, -1, 3, 2};
  int result = p3a::transform_reduce(
      p3a::execution::seq,
      test_data.begin(), test_data.end(),
      0, p3a::adder<int>(), p3a::identity<int>());
  EXPECT_EQ(result, 5);
}
