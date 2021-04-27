#pragma once

#include "p3a_reduce.hpp"

namespace p3a {

class conjugate_gradient_solver {
  device_array<double> m_r;
  device_array<double> m_p;
  device_array<double> m_Ap;
 public:
  using Ax_functor_type = std::function<
    void(device_array<double> const&, device_array<double>&)>;
  using b_functor_type = std::function<
    void(device_array<double>&)>;
  int solve(
      mpi::comm& comm,
      reproducible_floating_point_adder& adder,
      Ax_functor_type const& Ax_functor,
      b_functor_type const& b_functor,
      device_array<double>& x,
      double relative_tolerance);
};

}
