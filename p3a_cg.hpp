#pragma once

#include "p3a_dynamic_array.hpp"
#include "p3a_reduce.hpp"

namespace p3a {

template <
  class T,
  class Allocator = allocator<T>,
  class ExecutionPolicy = serial_execution>
class conjugate_gradient {
 public:
  using array_type = dynamic_array<T, Allocator, ExecutionPolicy>;
 private:
  array_type m_r;
  array_type m_p;
  array_type m_Ap;
  reproducible_adder<T, Allocator, ExecutionPolicy> m_adder;
 public:
  using A_action_type = std::function<
    void(array_type const&, array_type&)>;
  using b_filler_type = std::function<
    void(array_type&)>;
  conjugate_gradient() = default;
  conjugate_gradient(mpi::comm&& comm_arg)
    :m_adder(std::move(comm_arg))
  {}
  P3A_NEVER_INLINE int solve(
      A_action_type const& A_action,
      b_filler_type const& b_filler,
      array_type& x,
      T const& relative_tolerance);
};

template <
  class T,
  class Allocator,
  class ExecutionPolicy>
P3A_NEVER_INLINE double dot_product(
    reproducible_adder<T, Allocator, ExecutionPolicy>& adder,
    dynamic_array<T, Allocator, ExecutionPolicy> const& a,
    dynamic_array<T, Allocator, ExecutionPolicy> const& b)
{
  using size_type = typename dynamic_array<T, Allocator, ExecutionPolicy>::size_type;
  auto const a_ptr = a.cbegin();
  auto const b_ptr = b.cbegin();
  return adder.transform_reduce(
      counting_iterator<size_type>(0),
      counting_iterator<size_type>(a.size()),
  [=] P3A_HOST P3A_DEVICE (size_type i) P3A_ALWAYS_INLINE {
    return a_ptr[i] * b_ptr[i];
  });
}

template <
  class T,
  class Allocator,
  class ExecutionPolicy>
P3A_NEVER_INLINE void axpy(
    T a,
    dynamic_array<T, Allocator, ExecutionPolicy> const& x,
    dynamic_array<T, Allocator, ExecutionPolicy> const& y,
    dynamic_array<T, Allocator, ExecutionPolicy>& result)
{
  using size_type = typename dynamic_array<T, Allocator, ExecutionPolicy>::size_type;
  auto const x_ptr = x.cbegin();
  auto const y_ptr = y.cbegin();
  auto const result_ptr = result.begin();
  for_each(x.get_execution_policy(),
      counting_iterator<size_type>(0),
      counting_iterator<size_type>(x.size()),
  [=] P3A_HOST P3A_DEVICE (size_type i) P3A_ALWAYS_INLINE {
    result_ptr[i] = a * x_ptr[i] + y_ptr[i];
  });
}

template <
  class T,
  class Allocator,
  class ExecutionPolicy>
P3A_NEVER_INLINE int conjugate_gradient<T, Allocator, ExecutionPolicy>::solve(
      A_action_type const& A_action,
      b_filler_type const& b_filler,
      array_type& x,
      T const& relative_tolerance)
{
  this->m_r.resize(x.size());
  this->m_p.resize(x.size());
  this->m_Ap.resize(x.size());
  array_type& r = this->m_r;
  array_type& p = this->m_p;
  array_type& Ap = this->m_Ap;
  array_type& Ax = this->m_r;
  array_type& b = this->m_Ap;
  b_filler(b);
  T const b_dot_b = dot_product(m_adder, b, b);
  T const b_magnitude = square_root(b_dot_b);
  T const absolute_tolerance = b_magnitude * relative_tolerance;
  A_action(x, Ax);
  axpy(T(-1), Ax, b, r); // r = A * x - b
  copy(device, r.cbegin(), r.cend(), p.begin()); // p = r
  T r_dot_r_old = dot_product(m_adder, r, r);
  T residual_magnitude = square_root(r_dot_r_old);
  if (residual_magnitude <= absolute_tolerance) return 0;
  for (int k = 1; true; ++k) {
    A_action(p, Ap);
    T const pAp = dot_product(m_adder, p, Ap);
    T const alpha = r_dot_r_old / pAp; // alpha = (r^T * r) / (p^T * A * p)
    axpy(alpha, p, x, x); // x = x + alpha * p
    axpy(-alpha, Ap, r, r); // r = r - alpha * (A * p)
    T const r_dot_r_new = dot_product(m_adder, r, r);
    residual_magnitude = square_root(r_dot_r_old);
    if (residual_magnitude <= absolute_tolerance) {
      return k;
    }
    T const beta = r_dot_r_new / r_dot_r_old;
    axpy(beta, p, r, p); // p = r + beta * p
    r_dot_r_old = r_dot_r_new;
  }
}

}
