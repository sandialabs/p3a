#include "gtest/gtest.h"
#include "p3a_quantity.hpp"

TEST(quantity, divide) {
  auto a = p3a::meters<double>(1.0) / p3a::seconds<double>(2.0);
  static_assert(std::is_same_v<decltype(a), p3a::meters_per_second<double>>,
      "meters times seconds should be meters per second");
}
