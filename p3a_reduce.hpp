#pragma once

#include "mpicpp.hpp"

#include "p3a_execution.hpp"
#include "p3a_grid3.hpp"
#include "p3a_dynamic_array.hpp"
#include "p3a_counting_iterator.hpp"
#include "p3a_functional.hpp"

namespace p3a {

namespace details {

template <
  class ExecutionSpace,
  class Iterator,
  class T,
  class BinaryReductionOp,
  class UnaryTransformOp>
[[nodiscard]]
T kokkos_transform_reduce(
    Iterator first, Iterator last,
    T init,
    BinaryReductionOp binary_op,
    UnaryTransformOp unary_op)
{
  auto n = last - first;
  using difference_type = decltype(n);
  return kokkos_transform_reduce(
      p3a::counting_iterator<difference_type>(0),
      p3a::counting_iterator<difference_type>(n),
      init,
      binary_op,
}

}

namespace details {

class int128 {
  std::int64_t m_high;
  std::uint64_t m_low;
 public:
  P3A_ALWAYS_INLINE inline int128() = default;
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
  int128(std::int64_t high_arg, std::uint64_t low_arg)
    :m_high(high_arg)
    ,m_low(low_arg)
  {}
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
  int128(std::int64_t value)
    :int128(
        std::int64_t(-1) * (value < 0),
        std::uint64_t(value))
  {}
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
  std::int64_t high() const { return m_high; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
  std::uint64_t low() const { return m_low; }
};

template <
  class Allocator,
  class ExecutionPolicy>
class fixed_point_double_sum {
 public:
  using values_type = dynamic_array<double, Allocator, ExecutionPolicy>;
 private:
  mpicpp::comm m_comm;
  values_type m_values;
  reducer<int, ExecutionPolicy> m_exponent_reducer;
  reducer<int128, ExecutionPolicy> m_int128_reducer;
 public:
  fixed_point_double_sum() = default;
  explicit fixed_point_double_sum(mpicpp::comm&& comm_arg)
    :m_comm(std::move(comm_arg))
  {}
  fixed_point_double_sum(fixed_point_double_sum&&) = default;
  fixed_point_double_sum& operator=(fixed_point_double_sum&&) = default;
  fixed_point_double_sum(fixed_point_double_sum const&) = delete;
  fixed_point_double_sum& operator=(fixed_point_double_sum const&) = delete;
 public:
  [[nodiscard]] P3A_NEVER_INLINE
  double compute();
  [[nodiscard]] P3A_ALWAYS_INLINE inline constexpr
  values_type& values() { return m_values; }
};

extern template class fixed_point_double_sum<allocator<double>, serial_execution>;
#ifdef __CUDACC__
extern template class fixed_point_double_sum<cuda_device_allocator<double>, cuda_execution>;
#endif
#ifdef __HIPCC__
extern template class fixed_point_double_sum<hip_device_allocator<double>, hip_execution>;
#endif

}

template <class T, class Allocator, class ExecutionPolicy>
class associative_sum;

namespace details {

template <class Iterator, class SizeType, class UnaryOp>
class associative_sum_iterator_functor {
  Iterator first;
  double* values;
  UnaryOp unary_op;
 public:
  associative_sum_iterator_functor(
      Iterator first_arg,
      double* values_arg,
      UnaryOp unary_op_arg)
    :first(first_arg)
    ,values(values_arg)
    ,unary_op(unary_op_arg)
  {}
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE void operator()(SizeType i) const {
    values[i] = unary_op(first[i]);
  }
};

template <class UnaryOp>
class associative_sum_subgrid_functor {
  subgrid3 grid;
  double* values;
  UnaryOp unary_op;
 public:
  associative_sum_subgrid_functor(
      subgrid3 grid_arg,
      double* values_arg,
      UnaryOp unary_op_arg)
    :grid(grid_arg)
    ,values(values_arg)
    ,unary_op(unary_op_arg)
  {}
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE void operator()(vector3<int> const& grid_point) const {
    int const index = grid.index(grid_point);
    values[index] = unary_op(grid_point);
  }
};

}

template <
  class Allocator,
  class ExecutionPolicy>
class associative_sum<double, Allocator, ExecutionPolicy> {
  details::fixed_point_double_sum<Allocator, ExecutionPolicy> m_fixed_point;
 public:
  associative_sum() = default;
  explicit associative_sum(mpicpp::comm&& comm_arg)
    :m_fixed_point(std::move(comm_arg))
  {}
  associative_sum(associative_sum&&) = default;
  associative_sum& operator=(associative_sum&&) = default;
  associative_sum(associative_sum const&) = delete;
  associative_sum& operator=(associative_sum const&) = delete;
#ifdef __CUDACC__
 public:
#else
 private:
#endif
 public:
  template <class Iterator, class UnaryOp>
  [[nodiscard]]
  double transform_reduce(
      Iterator first,
      Iterator last,
      UnaryOp unary_op)
  {
    auto const n = (last - first);
    m_fixed_point.values().resize(n);
    auto const policy = m_fixed_point.values().get_execution_policy();
    auto const values = m_fixed_point.values().begin();
    using size_type = std::remove_const_t<decltype(n)>;
    for_each(policy,
    counting_iterator<size_type>(0),
    counting_iterator<size_type>(n),
    details::associative_sum_iterator_functor<Iterator, size_type, UnaryOp>(first, values, unary_op));
    return m_fixed_point.compute();
  }
  template <class UnaryOp>
  [[nodiscard]]
  double transform_reduce(
      subgrid3 grid,
      UnaryOp unary_op)
  {
    m_fixed_point.values().resize(grid.size());
    auto const policy = m_fixed_point.values().get_execution_policy();
    auto const values = m_fixed_point.values().begin();
    for_each(policy, grid, details::associative_sum_subgrid_functor<UnaryOp>(grid, values, unary_op));
    return m_fixed_point.compute();
  }
};

template <class T>
using device_associative_sum = 
  associative_sum<T, device_allocator<T>, device_execution>;
template <class T>
using host_associative_sum = 
  associative_sum<T, allocator<T>, serial_execution>;

}
