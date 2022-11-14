#include <gtest/gtest.h>

#include "p3a_search.hpp"

inline void search_invert_linear()
{
  auto const state_from_domain_value = [] P3A_HOST_DEVICE (double x) -> double {
    return x;
  };
  auto const range_value_from_state = [] P3A_HOST_DEVICE (double x) -> double {
    return x;
  };
  auto const derivative_value_from_state = [] P3A_HOST_DEVICE (double) -> double { return 1.0; };
  double const desired_range_value = 0.3;
  double const tolerance = 1.0e-6;
  double const minimum_domain_value = 0.0;
  double const maximum_domain_value = 1.0;
  double domain_value;
  auto const result = p3a::invert_differentiable_function(
      state_from_domain_value,
      range_value_from_state,
      derivative_value_from_state,
      desired_range_value,
      tolerance,
      minimum_domain_value,
      maximum_domain_value,
      domain_value);
  EXPECT_EQ(result, p3a::search_errc::success);
  EXPECT_FLOAT_EQ(domain_value, desired_range_value); // because linear
  EXPECT_FLOAT_EQ(range_value_from_state(state_from_domain_value(domain_value)), desired_range_value);
}

TEST(search, invert_linear)
{
  search_invert_linear();
}

inline void search_invert_cosine()
{
  auto const state_from_domain_value = [] P3A_HOST_DEVICE (double x) -> double {
    return x;
  };
  auto const range_value_from_state = [] P3A_HOST_DEVICE (double x) -> double {
    auto const result = std::cos(x);
    return result;
  };
  auto const derivative_value_from_state = [] P3A_HOST_DEVICE (double x) -> double {
    auto const result = -std::sin(x);
    return result;
  };
  double const desired_range_value = 0.3;
  double const tolerance = 1.0e-6;
  double const minimum_domain_value = 0.0;
  double const maximum_domain_value = p3a::pi_value<double>();
  double domain_value;
  auto const result = p3a::invert_differentiable_function(
      state_from_domain_value,
      range_value_from_state,
      derivative_value_from_state,
      desired_range_value,
      tolerance,
      minimum_domain_value,
      maximum_domain_value,
      domain_value);
  EXPECT_EQ(result, p3a::search_errc::success);
  EXPECT_FLOAT_EQ(range_value_from_state(state_from_domain_value(domain_value)), desired_range_value);
}

// the point of this test is to have an input where the derivative
// of the function is zero at both endpoints of the subset of the
// domain being searched.
// In this case, Newton's method should not be enough by itself.
TEST(search, invert_cosine)
{
  search_invert_cosine();
}

inline void search_invert_non_monotonic()
{
  auto const state_from_domain_value = [] P3A_HOST_DEVICE (double x) -> double {
    return x;
  };
  auto const range_value_from_state = [] P3A_HOST_DEVICE (double x) -> double {
    auto const result = std::sin(x);
    return result;
  };
  auto const derivative_value_from_state = [] P3A_HOST_DEVICE (double x) -> double {
    auto const result = std::cos(x);
    return result;
  };
  double const desired_range_value = 0.3;
  double const tolerance = 1.0e-6;
  double const minimum_domain_value = (1.0 / 4.0) * p3a::pi_value<double>();
  double const maximum_domain_value = (7.0 / 4.0) * p3a::pi_value<double>();
  double domain_value;
  auto const result = p3a::invert_differentiable_function(
      state_from_domain_value,
      range_value_from_state,
      derivative_value_from_state,
      desired_range_value,
      tolerance,
      minimum_domain_value,
      maximum_domain_value,
      domain_value);
  EXPECT_EQ(result, p3a::search_errc::success);
  EXPECT_FLOAT_EQ(range_value_from_state(state_from_domain_value(domain_value)), desired_range_value);
}

// the point of this test is to have an input where the function
// is not monotonic in the subset of the domain specified.
// in fact, Newton's method starting at an endpoint will try to
// leave the subset of the domain in this case.
TEST(search, invert_non_monotonic)
{
  search_invert_non_monotonic();
}

TEST(search, tabulated_interval)
{
  int constexpr n = 4;
  double const table[n] = {1.0, 2.0, 3.0, 4.0};
  int i;
  auto result = p3a::find_tabulated_interval(n, p3a::iterator_as_functor(table), 2.5, i);
  EXPECT_EQ(result, p3a::search_errc::success);
  EXPECT_EQ(i, 1);
  result = p3a::find_tabulated_interval(n, p3a::iterator_as_functor(table), 1.0, i);
  EXPECT_EQ(result, p3a::search_errc::success);
  EXPECT_EQ(i, 0);
  result = p3a::find_tabulated_interval(n, p3a::iterator_as_functor(table), 4.0, i);
  EXPECT_EQ(result, p3a::search_errc::success);
  EXPECT_EQ(i, 2);
}

TEST(search, cos_no_deriv)
{
  auto const function =
  [] P3A_HOST_DEVICE (double x)
  {
    return std::sin(x);
  };
  double const domain_lower_bound = 0.0;
  double const domain_upper_bound = 1.5;
  double const desired_range_value = 0.9;
  double domain_value;
  double const tolerance = 1.0e-6;
  int const maximum_iterations = 100;
  auto const error = p3a::invert_function(
      function,
      desired_range_value,
      domain_value,
      domain_lower_bound,
      domain_upper_bound,
      tolerance,
      maximum_iterations);
  EXPECT_EQ(error, p3a::search_errc::success);
  double const range_value = function(domain_value);
  EXPECT_NEAR(range_value, desired_range_value, tolerance);
}
