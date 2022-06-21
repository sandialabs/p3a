#include <gtest/gtest.h>

#include "p3a_search.hpp"

TEST(search, invert_linear)
{
  int evaluation_counter = 0;
  auto const state_from_domain_value = [&] (double x) -> double {
    ++evaluation_counter;
    return x;
  };
  auto const range_value_from_state = [] (double x) -> double {
    return x;
  };
  auto const derivative_value_from_state = [] (double) -> double { return 1.0; };
  double const desired_range_value = 0.3;
  double const tolerance = 1.0e-6;
  double const minimum_domain_value = 0.0;
  double const maximum_domain_value = 1.0;
  auto const minimum_domain_state = state_from_domain_value(minimum_domain_value);
  auto const range_at_minimum_domain_value = range_value_from_state(minimum_domain_state);
  auto const range_at_maximum_domain_value = range_value_from_state(state_from_domain_value(maximum_domain_value));
  auto domain_value = minimum_domain_value;
  auto range_value = range_at_minimum_domain_value;
  auto derivative_value = derivative_value_from_state(minimum_domain_state);
  p3a::invert_differentiable_function(
      state_from_domain_value,
      range_value_from_state,
      derivative_value_from_state,
      desired_range_value,
      tolerance,
      minimum_domain_value,
      maximum_domain_value,
      range_at_minimum_domain_value,
      range_at_maximum_domain_value,
      domain_value,
      range_value,
      derivative_value);
  EXPECT_FLOAT_EQ(range_value, desired_range_value);
  EXPECT_FLOAT_EQ(domain_value, desired_range_value); // because linear
  EXPECT_EQ(evaluation_counter, 3);
}

// the point of this test is to have an input where the derivative
// of the function is zero at both endpoints of the subset of the
// domain being searched.
// In this case, Newton's method should not be enough by itself.
TEST(search, invert_cosine)
{
  int evaluation_counter = 0;
  auto const state_from_domain_value = [&] (double x) -> double {
    ++evaluation_counter;
    return x;
  };
  auto const range_value_from_state = [] (double x) -> double {
    auto const result = std::cos(x);
    return result;
  };
  auto const derivative_value_from_state = [] (double x) -> double {
    auto const result = -std::sin(x);
    return result;
  };
  double const desired_range_value = 0.3;
  double const tolerance = 1.0e-6;
  double const minimum_domain_value = 0.0;
  double const maximum_domain_value = p3a::pi_value<double>();
  auto const minimum_domain_state = state_from_domain_value(minimum_domain_value);
  auto const range_at_minimum_domain_value = range_value_from_state(minimum_domain_state);
  auto const range_at_maximum_domain_value = range_value_from_state(state_from_domain_value(maximum_domain_value));
  auto domain_value = minimum_domain_value;
  auto range_value = range_at_minimum_domain_value;
  auto derivative_value = derivative_value_from_state(minimum_domain_state);
  p3a::invert_differentiable_function(
      state_from_domain_value,
      range_value_from_state,
      derivative_value_from_state,
      desired_range_value,
      tolerance,
      minimum_domain_value,
      maximum_domain_value,
      range_at_minimum_domain_value,
      range_at_maximum_domain_value,
      domain_value,
      range_value,
      derivative_value);
  EXPECT_FLOAT_EQ(range_value, desired_range_value);
}

// the point of this test is to have an input where the function
// is not monotonic in the subset of the domain specified.
// in fact, Newton's method starting at an endpoint will try to
// leave the subset of the domain in this case.
TEST(search, invert_non_monotonic)
{
  int evaluation_counter = 0;
  auto const state_from_domain_value = [&] (double x) -> double {
    ++evaluation_counter;
    return x;
  };
  auto const range_value_from_state = [] (double x) -> double {
    auto const result = std::sin(x);
    return result;
  };
  auto const derivative_value_from_state = [] (double x) -> double {
    auto const result = std::cos(x);
    return result;
  };
  double const desired_range_value = 0.3;
  double const tolerance = 1.0e-6;
  double const minimum_domain_value = (1.0 / 4.0) * p3a::pi_value<double>();
  double const maximum_domain_value = (7.0 / 4.0) * p3a::pi_value<double>();
  auto const minimum_domain_state = state_from_domain_value(minimum_domain_value);
  auto const range_at_minimum_domain_value = range_value_from_state(minimum_domain_state);
  auto const range_at_maximum_domain_value = range_value_from_state(state_from_domain_value(maximum_domain_value));
  auto domain_value = minimum_domain_value;
  auto range_value = range_at_minimum_domain_value;
  auto derivative_value = derivative_value_from_state(minimum_domain_state);
  p3a::invert_differentiable_function(
      state_from_domain_value,
      range_value_from_state,
      derivative_value_from_state,
      desired_range_value,
      tolerance,
      minimum_domain_value,
      maximum_domain_value,
      range_at_minimum_domain_value,
      range_at_maximum_domain_value,
      domain_value,
      range_value,
      derivative_value);
  EXPECT_FLOAT_EQ(range_value, desired_range_value);
}
