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
P3A_HOST_DEVICE inline
void invert_differentiable_function(
    StateFromDomainValue const& state_from_domain_value,
    RangeValueFromState const& range_value_from_state,
    DerivativeValueFromState const& derivative_value_from_state,
    RangeValue const& desired_range_value,
    Tolerance const& tolerance,
    DomainValue minimum_domain_value,
    DomainValue maximum_domain_value,
    RangeValue range_value_at_minimum_domain_value,
    RangeValue range_value_at_maximum_domain_value,
    DomainValue& domain_value,
    RangeValue& range_value,
    DerivativeValue& derivative_value)
{
  while (true) {
    if (are_close(range_value, desired_range_value, tolerance)) return;
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
    auto const next_state = state_from_domain_value(domain_value);
    range_value = range_value_from_state(next_state);
    derivative_value = range_value_from_state(next_state);
    auto const next_is_less = (range_value < desired_range_value);
    minimum_domain_value = condition(next_is_less, domain_value, minimum_domain_value);
    maximum_domain_value = condition(next_is_less, maximum_domain_value, domain_value);
    range_value_at_minimum_domain_value = condition(next_is_less,
        range_value, range_value_at_minimum_domain_value);
    range_value_at_maximum_domain_value = condition(next_is_less,
        range_value_at_maximum_domain_value, range_value);
  }
}

}
