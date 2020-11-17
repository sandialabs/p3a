#pragma once

namespace p3a {

template <class T>
class minimizer {
 public:
  CPL_ALWAYS_INLINE constexpr
  T operator()(T const& a, T const& b) const {
    return minimum(a, b);
  }
};
template <class T> inline constexpr minimizer<T> minimizes = {};

template <class T>
class adder {
 public:
  CPL_ALWAYS_INLINE constexpr
  T operator()(T const& a, T const& b) const {
    return a + b;
  }
};
template <class T> inline constexpr adder<T> adds = {};

template <
    class T,
    class BinaryOp,
    class UnaryOp>
T transform_reduce(
    execution::serial_policy,
    subgrid3 subgrid,
    T init,
    BinaryOp binary_op,
    UnaryOp unary_op)
{
  for (int k = subgrid.lower().z(); k < subgrid.upper().z(); ++k) {
    for (int j = subgrid.lower().y(); j < subgrid.upper().y(); ++j) {
      for (int i = subgrid.lower().x(); i < subgrid.upper().x(); ++i) {
        init = binary_op(std::move(init),
            unary_op(vector3<int>(i, j, k)));
      }
    }
  }
  return init;
}

}
