#pragma once

#include "p3a_reduce.hpp"

namespace p3a {

class conjugate_gradient_solver {
  device_array<double> m_r;
  device_array<double> m_p;
  device_array<double> m_Ap;
  reproducible_floating_point_adder m_adder;
 public:
  using Ax_functor_type = std::function<
    void(device_array<double> const&, device_array<double>&)>;
  using b_functor_type = std::function<
    void(device_array<double>&)>;
  conjugate_gradient_solver() = default;
  conjugate_gradient_solver(mpi::comm&& comm_arg)
    :m_adder(std::move(comm_arg))
  {}
  int solve(
      Ax_functor_type const& Ax_functor,
      b_functor_type const& b_functor,
      device_array<double>& x,
      double relative_tolerance);
};

}
