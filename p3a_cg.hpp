#pragma once

#include <stdexcept>
#include <string>

#include "p3a_dynamic_array.hpp"
#include "p3a_reduce.hpp"

namespace p3a {

class convergence_failure : public std::runtime_error {
 public:
  convergence_failure(std::string const& msg)
    :std::runtime_error(msg)
  {
  }
};

template <
  class T,
  class Allocator = host_allocator<T>,
  class ExecutionPolicy = execution::sequenced_policy>
class preconditioned_conjugate_gradient {
 public:
  using array_type = dynamic_array<T, Allocator, ExecutionPolicy>;
 private:
  array_type m_r;
  array_type m_z;
  array_type m_p;
  array_type m_scratch;
  associative_sum<T, Allocator, ExecutionPolicy> m_adder;
  T m_relative_tolerance = 1.0e-6;
  int m_maximum_iterations = 1'000'000;
 public:
  using M_inv_action_type = std::function<
    void(array_type const&, array_type&)>;
  using A_action_type = std::function<
    void(array_type const&, array_type&)>;
  using b_filler_type = std::function<
    void(array_type&)>;
  preconditioned_conjugate_gradient() = default;
  preconditioned_conjugate_gradient(mpicpp::comm&& comm_arg)
    :m_adder(std::move(comm_arg))
  {}
  void set_relative_tolerance(T const& arg)
  {
    m_relative_tolerance = arg;
  }
  void set_maximum_iterations(int arg)
  {
    m_maximum_iterations = arg;
  }
  P3A_NEVER_INLINE int solve(
      M_inv_action_type const& M_inv_action,
      A_action_type const& A_action,
      b_filler_type const& b_filler,
      array_type& x);
};

template <
  class T,
  class Allocator,
  class ExecutionPolicy>
P3A_NEVER_INLINE double dot_product(
    associative_sum<T, Allocator, ExecutionPolicy>& adder,
    dynamic_array<T, Allocator, ExecutionPolicy> const& a,
    dynamic_array<T, Allocator, ExecutionPolicy> const& b)
{
  using size_type = typename dynamic_array<T, Allocator, ExecutionPolicy>::size_type;
  auto const a_ptr = a.cbegin();
  auto const b_ptr = b.cbegin();
  return adder.transform_reduce(
      counting_iterator<size_type>(0),
      counting_iterator<size_type>(a.size()),
  [=] P3A_HOST_DEVICE (size_type i) P3A_ALWAYS_INLINE {
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
  [=] P3A_HOST_DEVICE (size_type i) P3A_ALWAYS_INLINE {
    result_ptr[i] = a * x_ptr[i] + y_ptr[i];
  });
}

template <
  class T,
  class Allocator,
  class ExecutionPolicy>
P3A_NEVER_INLINE
int preconditioned_conjugate_gradient<T, Allocator, ExecutionPolicy>::solve(
      M_inv_action_type const& M_inv_action,
      A_action_type const& A_action,
      b_filler_type const& b_filler,
      array_type& x)
{
  this->m_r.resize(x.size());
  this->m_z.resize(x.size());
  this->m_p.resize(x.size());
  this->m_scratch.resize(x.size());
  array_type& r = this->m_r;
  array_type& z = this->m_z;
  array_type& p = this->m_p;
  array_type& b = this->m_scratch;
  array_type& Ap = this->m_scratch;
  array_type& Ax = this->m_r;
  b_filler(b);
  T const b_dot_b = dot_product(m_adder, b, b);
  T const b_magnitude = p3a::sqrt(b_dot_b);
  if (b_magnitude == T(0)) {
    throw std::invalid_argument("P3A CG solver: the magnitude of the right hand side vector is zero");
  }
  T const absolute_tolerance = b_magnitude * m_relative_tolerance;
  A_action(x, Ax); // Ax = A * x
  axpy(T(-1), Ax, b, r); // r = A * x - b
  T residual_magnitude = p3a::sqrt(dot_product(m_adder, r, r));
  if (residual_magnitude <= absolute_tolerance) return 0;
  M_inv_action(r, z);  // z = M^-1 * r
  T r_dot_z_old = dot_product(m_adder, r, z); // r^T * z
  copy(p.get_execution_policy(), z.cbegin(), z.cend(), p.begin()); // p = z
  for (int k = 1; true; ++k) {
    A_action(p, Ap);
    T const pAp = dot_product(m_adder, p, Ap);
    T const alpha = r_dot_z_old / pAp; // alpha = (r^T * z) / (p^T * A * p)
    axpy(alpha, p, x, x); // x = x + alpha * p
    axpy(-alpha, Ap, r, r); // r = r - alpha * (A * p)
    residual_magnitude = p3a::sqrt(dot_product(m_adder, r, r));
    if (residual_magnitude <= absolute_tolerance) {
      return k;
    }
    if (k == m_maximum_iterations) {
      throw convergence_failure(
          "P3A CG solver failed to converge in " + std::to_string(m_maximum_iterations) +
          " iterations. relative tolerance was " + std::to_string(m_relative_tolerance) +
          ", right hand side magnitude was " + std::to_string(b_magnitude) +
          ", absolute_tolerance was " + std::to_string(absolute_tolerance) +
          ", and final residual magnitude was " + std::to_string(residual_magnitude));
    }
    M_inv_action(r, z); // z = M^-1 r
    T const r_dot_z_new = dot_product(m_adder, r, z);
    T const beta = r_dot_z_new / r_dot_z_old;
    axpy(beta, p, z, p); // p = z + beta * p;
    r_dot_z_old = r_dot_z_new;
  }
}

}
