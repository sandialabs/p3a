#include "p3a_cg.hpp"

#include "p3a_counting_iterator.hpp"

namespace p3a {

double dot_product(
    mpi::comm& comm,
    reproducible_floating_point_adder& adder,
    device_array<double> const& a,
    device_array<double> const& b)
{
  using size_type = device_array<double>::size_type;
  auto const a_ptr = a.cbegin();
  auto const b_ptr = b.cbegin();
  return adder.transform_reduce(
      comm,
      counting_iterator<size_type>(0),
      counting_iterator<size_type>(a.size()),
  [=] P3A_DEVICE (size_type i) P3A_ALWAYS_INLINE {
    return a_ptr[i] * b_ptr[i];
  });
}

void axpy(
    double a,
    device_array<double> const& x,
    device_array<double> const& y,
    device_array<double>& result)
{
  using size_type = device_array<double>::size_type;
  auto const x_ptr = x.cbegin();
  auto const y_ptr = y.cbegin();
  auto const result_ptr = result.begin();
  for_each(device,
      counting_iterator<size_type>(0),
      counting_iterator<size_type>(x.size()),
  [=] P3A_DEVICE (size_type i) P3A_ALWAYS_INLINE {
    result_ptr[i] = a * x_ptr[i] + y_ptr[i];
  });
}

int conjugate_gradient_solver::solve(
      mpi::comm& comm,
      reproducible_floating_point_adder& adder,
      Ax_functor_type const& A_action,
      b_functor_type const& b_functor,
      device_array<double>& x,
      double relative_tolerance)
{
  device_array<double>& r = this->m_r;
  device_array<double>& p = this->m_p;
  device_array<double>& Ap = this->m_Ap;
  device_array<double>& Ax = this->m_r;
  device_array<double>& b = this->m_Ap;
  b_functor(b);
  double const b_dot_b = dot_product(comm, adder, b, b);
  double const b_magnitude = square_root(b_dot_b);
  double const absolute_tolerance = b_magnitude * relative_tolerance;
  A_action(x, Ax);
  axpy(-1.0, Ax, b, r); // r = A * x - b
  copy(device, r.cbegin(), r.cend(), p.begin()); // p = r
  double r_dot_r_old = dot_product(comm, adder, r, r);
  double residual_magnitude = square_root(r_dot_r_old);
  if (residual_magnitude <= absolute_tolerance) return 0;
  for (int k = 1; true; ++k) {
    A_action(p, Ap);
    double const pAp = dot_product(comm, adder, p, Ap);
    double const alpha = r_dot_r_old / pAp; // alpha = (r^T * r) / (p^T * A * p)
    axpy(alpha, p, x, x); // x = x + alpha * p
    axpy(-alpha, Ap, r, r); // r = r - alpha * (A * p)
    double const r_dot_r_new = dot_product(comm, adder, r, r);
    residual_magnitude = square_root(r_dot_r_old);
    if (residual_magnitude <= absolute_tolerance) {
      return k;
    }
    double const beta = r_dot_r_new / r_dot_r_old;
    axpy(beta, p, r, p); // p = r + beta * p
    r_dot_r_old = r_dot_r_new;
  }
}

}
