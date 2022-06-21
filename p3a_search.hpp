#pragma once

#include "p3a_macros.hpp"
#include "p3a_functions.hpp"

namespace p3a {

template <class T = void>
class less {
 public:
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  auto operator()(T const& lhs, T const& rhs ) const
  {
    return lhs < rhs;
  }
};

template <class ForwardIt, class T, class Compare>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline
ForwardIt upper_bound(
    ForwardIt first,
    ForwardIt last,
    T const& value,
    Compare comp)
{
  auto count = last - first;
  while (count > 0) {
    auto it = first; 
    auto const step = count / 2;
    it += step;
    if (!comp(value, *it)) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first;
}

template<class ForwardIt, class T, class Compare>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline
ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value, Compare comp)
{
  auto count = last - first;
  while (count > 0) {
    auto it = first;
    auto const step = count / 2;
    it += step;
    if (comp(*it, value)) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first;
}

template <class ForwardIt, class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline
ForwardIt upper_bound(
    ForwardIt first,
    ForwardIt last,
    T const& value)
{
  return p3a::upper_bound(first, last, value, p3a::less<T>());
}

template <class ForwardIt, class T>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline
ForwardIt lower_bound(
    ForwardIt first,
    ForwardIt last,
    T const& value)
{
  return p3a::lower_bound(first, last, value, p3a::less<T>());
}

enum class search_errc : int {
  success,
  desired_value_below_minimum,
  desired_value_above_maximum,
  exceeded_maximum_iterations
};

/* This code acts as the inverse of a real differentiable function
 * in a specified subset of the domain.
 * It is given the ability to compute the function and its first derivative.
 * It then uses a combination of methods to search the given space until it finds
 * a domain value for which the range value is close enough to the desired range value.
 * The primary method is Newton's method.
 * In cases where Newton's method on the real function would not converge,
 * we fall back to using Newton's method on a linear approximation to
 * the real function (we assume the function is linear between the endpoints
 * of the subset of the domain).
 *
 * The function only needs to be continuous and differentiable, it does not need
 * to be monotonic in the given subset of the domain.
 * 
 * Execution speed via minimizing actual function evaluations is a
 * primary design goal of this code.
 * This is the reason for using Newton's method in the common case
 * to do fewer evaluations than something like bisection would.
 * This is also the reason for having a separate "function state"
 * that the value and derivative are computed from.
 * This is because the derivative calculation can often use information
 * from the value calculation, so "function state" is a mechanism
 * for users to implement that optimization.
 */

template <
  class DomainValue,
  class RangeValue,
  class DerivativeValue,
  class Tolerance,
  class StateFromDomainValue,
  class RangeValueFromState,
  class DerivativeValueFromState>
[[nodiscard]] P3A_HOST_DEVICE inline
search_errc invert_differentiable_function(
    StateFromDomainValue const& state_from_domain_value,
    RangeValueFromState const& range_value_from_state,
    DerivativeValueFromState const& derivative_value_from_state,
    RangeValue const& desired_range_value,
    Tolerance const& tolerance,
    DomainValue minimum_domain_value,
    DomainValue maximum_domain_value,
    DomainValue& domain_value,
    RangeValue& range_value,
    DerivativeValue& derivative_value)
{
  int constexpr maximum_iterations = 100;
  auto const state_at_maximum_domain_value = state_from_domain_value(maximum_domain_value);
  auto range_value_at_maximum_domain_value = range_value_from_state(state_at_maximum_domain_value);
  domain_value = minimum_domain_value;
  auto state_at_domain_value = state_from_domain_value(domain_value);
  auto range_value_at_minimum_domain_value = range_value_from_state(state_at_domain_value);
  range_value = range_value_at_minimum_domain_value;
  derivative_value = derivative_value_from_state(state_at_domain_value);
  for (int iteration = 0; iteration < maximum_iterations; ++iteration) {
    if (are_close(range_value, desired_range_value, tolerance)) return search_errc::success;
    auto const next_domain_value_newton =
      domain_value - (range_value - desired_range_value) / derivative_value;
    auto const linear_derivative =
      (range_value_at_maximum_domain_value - range_value_at_minimum_domain_value) /
      (maximum_domain_value - minimum_domain_value);
    auto const next_domain_value_linear =
      clamp(
          minimum_domain_value + (desired_range_value - range_value_at_minimum_domain_value) / linear_derivative,
          minimum_domain_value,
          maximum_domain_value);
    auto const newton_will_not_converge =
      (derivative_value == DerivativeValue(0)) ||
      (next_domain_value_newton > maximum_domain_value) ||
      (next_domain_value_newton < minimum_domain_value);
    domain_value =
      condition(
          newton_will_not_converge,
          next_domain_value_linear,
          next_domain_value_newton);
    state_at_domain_value = state_from_domain_value(domain_value);
    range_value = range_value_from_state(state_at_domain_value);
    derivative_value = derivative_value_from_state(state_at_domain_value);
    auto const is_new_minimum = 
      // this is a logical XOR operation, designed to flip the logic if the function
      // is decreasing rather than increasing
      (!(range_value < desired_range_value)) !=
      (!(range_value_at_maximum_domain_value < range_value_at_minimum_domain_value)); 
    minimum_domain_value = condition(is_new_minimum, domain_value, minimum_domain_value);
    maximum_domain_value = condition(is_new_minimum, maximum_domain_value, domain_value);
    range_value_at_minimum_domain_value = condition(is_new_minimum,
        range_value, range_value_at_minimum_domain_value);
    range_value_at_maximum_domain_value = condition(is_new_minimum,
        range_value_at_maximum_domain_value, range_value);
  }
  return search_errc::exceeded_maximum_iterations;
}

/* given a set of tabulated values of a continuous real function,
 * this code finds an interval such that the tabulated range values
 * on either side of that interval bound the desired range value.
 *
 * It uses binary search (bisection in index space) to do this.
 */

template <
  class Index,
  class RangeValueFromPoint,
  class RangeValue>
[[nodiscard]] P3A_HOST_DEVICE inline
search_errc find_tabulated_interval(
    Index const& number_of_points,
    RangeValueFromPoint const& range_value_from_point,
    RangeValue const& desired_range_value,
    Index& interval)
{
  auto minimum_point = Index(0);
  auto maximum_point = number_of_points - 1;
  auto range_value_at_minimum_point = range_value_from_point(minimum_point);
  auto range_value_at_maximum_point = range_value_from_point(maximum_point);
  auto const minimum_range_value = min(range_value_at_minimum_point, range_value_at_maximum_point);
  auto const maximum_range_value = max(range_value_at_minimum_point, range_value_at_maximum_point);
  if (desired_range_value < minimum_range_value) return search_errc::desired_value_below_minimum;
  if (desired_range_value > maximum_range_value) return search_errc::desired_value_above_maximum;
  int constexpr maximum_iterations = 100;
  for (int iteration = 0; iteration < maximum_iterations; ++iteration) {
    if ((maximum_point - minimum_point) <= Index(1)) return search_errc::success;
    auto const point = minimum_point + (maximum_point - minimum_point) / 2;
    interval = point;
    auto const range_value = range_value_from_point(point);
    auto const is_new_minimum =
      (!(range_value < desired_range_value)) !=
      (!(range_value_at_maximum_point < range_value_at_minimum_point));
    minimum_point = condition(is_new_minimum, point, minimum_point);
    maximum_point = condition(is_new_minimum, maximum_point, point);
    range_value_at_minimum_point = condition(is_new_minimum,
        range_value, range_value_at_minimum_point);
    range_value_at_maximum_point = condition(is_new_minimum,
        range_value_at_maximum_point, range_value);
  }
  return search_errc::exceeded_maximum_iterations;
}

}
