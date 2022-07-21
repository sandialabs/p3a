#pragma once

#include "mpicpp.hpp"

#include "p3a_execution.hpp"
#include "p3a_grid3.hpp"
#include "p3a_dynamic_array.hpp"
#include "p3a_counting_iterator.hpp"
#include "p3a_functional.hpp"
#include "p3a_simd.hpp"

namespace p3a {

namespace details {

class int128 {
  std::int64_t m_high;
  std::uint64_t m_low;
 public:
  P3A_ALWAYS_INLINE inline int128() = default;
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  int128(std::int64_t high_arg, std::uint64_t low_arg)
    :m_high(high_arg)
    ,m_low(low_arg)
  {}
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  int128(std::int64_t value)
    :int128(
        std::int64_t(-1) * (value < 0),
        std::uint64_t(value))
  {}
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  std::int64_t high() const { return m_high; }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  std::uint64_t low() const { return m_low; }
};

}

// hack! in the fixed point reduction we assume that individual significands
// have at most 52 significant bits, so the sum of a small number of these
// (about 10) should not exceed 63 significant bits.
// this overload is a way to trick the system into first adding the 64-bit
// numbers and then converting to a 128-bit class
template <class Abi>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline
details::int128
reduce(
    const_where_expression<simd_mask<std::int64_t, Abi>, simd<std::int64_t, Abi>> const& we,
    details::int128 identity_value,
    adder<details::int128>)
{
  return details::int128(p3a::reduce(we, std::int64_t(0), p3a::adder<std::int64_t>()));
}
namespace details {

template <class T, class BinaryReductionOp>
class kokkos_reducer {
 public:
  using reducer = kokkos_reducer<T, BinaryReductionOp>;
  using value_type = T;
  using result_view_type = Kokkos::View<value_type, Kokkos::HostSpace>;
 private:
  value_type m_init;
  BinaryReductionOp m_binary_op;
  result_view_type m_result_view;
 public:
  kokkos_reducer(
      value_type init_arg,
      BinaryReductionOp binary_op_arg,
      value_type& result_arg)
    :m_init(init_arg)
    ,m_binary_op(binary_op_arg)
    ,m_result_view(&result_arg)
  {
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
  void join(value_type& dest, value_type const& src) const
  {
    dest = m_binary_op(dest, src);
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
  void init(value_type& val) const
  {
    val = m_init;
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
  value_type& reference() const
  {
    return *m_result_view.data();
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
  result_view_type view() const
  {
    return m_result_view;
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline constexpr
  bool references_scalar() const
  {
    return true;
  }
};

template <class T, class BinaryReductionOp, class UnaryTransformOp, class Integral>
class kokkos_reduce_functor {
  BinaryReductionOp m_binary_op;
  UnaryTransformOp m_unary_op;
 public:
  kokkos_reduce_functor(
      BinaryReductionOp binary_op_arg,
      UnaryTransformOp unary_op_arg)
    :m_binary_op(binary_op_arg)
    ,m_unary_op(unary_op_arg)
  {
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline void operator()(Integral i, T& updated) const
  {
    updated = m_binary_op(updated, m_unary_op(i));
  }
};

template <
  class ExecutionSpace,
  class Integral,
  class T,
  class BinaryReductionOp,
  class UnaryTransformOp>
[[nodiscard]]
std::enable_if_t<std::is_integral_v<Integral>, T>
kokkos_transform_reduce(
    p3a::counting_iterator<Integral> first,
    p3a::counting_iterator<Integral> last,
    T init,
    BinaryReductionOp binary_op,
    UnaryTransformOp unary_op)
{
  using functor = kokkos_reduce_functor<
    T, BinaryReductionOp, UnaryTransformOp, Integral>;
  using reducer_type = kokkos_reducer<T, BinaryReductionOp>;
  using kokkos_policy_type =
    Kokkos::RangePolicy<
      ExecutionSpace,
      Kokkos::IndexType<Integral>>;
  T result = init;
  reducer_type reducer(init, binary_op, result);
  Kokkos::parallel_reduce(
      "p3a::details::kokkos_transform_reduce(1D)",
      kokkos_policy_type(*first, *last),
      functor(binary_op, unary_op),
      reducer);
  return result;
}

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
  using difference_type = typename std::iterator_traits<Iterator>::difference_type;
  difference_type const n = last - first;
  using new_transform_type = kokkos_iterator_functor<Iterator, UnaryTransformOp>;
  return kokkos_transform_reduce(
      p3a::counting_iterator<difference_type>(0),
      p3a::counting_iterator<difference_type>(n),
      init,
      binary_op,
      new_transform_type(first, unary_op));
}

template <class T, class BinaryReductionOp, class UnaryTransformOp, class Integral>
class kokkos_3d_reduce_functor {
  BinaryReductionOp m_binary_op;
  UnaryTransformOp m_unary_op;
 public:
  kokkos_3d_reduce_functor(
      BinaryReductionOp binary_op_arg,
      UnaryTransformOp unary_op_arg)
    :m_binary_op(binary_op_arg)
    ,m_unary_op(unary_op_arg)
  {
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
  void operator()(Integral i, Integral j, Integral k, T& updated) const
  {
    updated = m_binary_op(updated, m_unary_op(i, j, k));
  }
};

template <
  class ExecutionSpace,
  class Integral,
  class T,
  class BinaryReductionOp,
  class UnaryTransformOp>
[[nodiscard]]
std::enable_if_t<std::is_integral_v<Integral>, T>
kokkos_transform_reduce(
    p3a::counting_iterator3<Integral> first,
    p3a::counting_iterator3<Integral> last,
    T init,
    BinaryReductionOp binary_op,
    UnaryTransformOp unary_op)
{
  using new_transform = kokkos_3d_functor<Integral, UnaryTransformOp>;
  using functor_type = kokkos_3d_reduce_functor<
    T, BinaryReductionOp, new_transform, Integral>;
  using reducer = kokkos_reducer<T, BinaryReductionOp>;
  using kokkos_policy =
    Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::IndexType<Integral>,
      Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>;
  T result = init;
  Kokkos::parallel_reduce(
      "p3a::details::kokkos_transform_reduce(3D)",
      kokkos_policy(
        {first.vector.x(), first.vector.y(), first.vector.z()},
        {last.vector.x(), last.vector.y(), last.vector.z()}),
      functor_type(binary_op, new_transform(unary_op)),
      reducer(init, binary_op, result));
  return result;
}

template <class T, class BinaryReductionOp, class UnaryTransformOp>
class simd_reduce_wrapper {
  T m_init;
  BinaryReductionOp m_binary_op;
  UnaryTransformOp m_unary_op;
 public:
  simd_reduce_wrapper(
      T init_arg,
      BinaryReductionOp binary_op_arg,
      UnaryTransformOp unary_op_arg)
    :m_init(init_arg)
    ,m_binary_op(binary_op_arg)
    ,m_unary_op(unary_op_arg)
  {
  }
  template <class Indices, class Abi>
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
  auto operator()(Indices const& indices, p3a::simd_mask<T, Abi> const& mask) const
  {
    auto const simd_result = m_unary_op(indices, mask);
    return p3a::reduce(where(mask, simd_result), m_init, m_binary_op);
  }
};

template <
  class SimdAbi,
  class ExecutionSpace,
  class Integral,
  class T,
  class BinaryReductionOp,
  class UnaryTransformOp>
[[nodiscard]]
std::enable_if_t<std::is_integral_v<Integral>, T>
kokkos_simd_transform_reduce(
    p3a::counting_iterator<Integral> first,
    p3a::counting_iterator<Integral> last,
    T init,
    BinaryReductionOp binary_op,
    UnaryTransformOp unary_op)
{
  Integral const extent = *last - *first;
  if (extent == 0) return init;
  using transform_a = simd_reduce_wrapper<T, BinaryReductionOp, UnaryTransformOp>;
  using transform_b = simd_functor<T, SimdAbi, Integral, transform_a>;
  using functor = kokkos_reduce_functor<
    T, BinaryReductionOp, transform_b, Integral>;
  using reducer = kokkos_reducer<T, BinaryReductionOp>;
  using kokkos_policy =
    Kokkos::RangePolicy<
      ExecutionSpace,
      Kokkos::IndexType<Integral>>;
  T result = init;
  Integral constexpr width = Integral(p3a::simd_mask<T, SimdAbi>::size());
  Integral const quotient = extent / width;
  Kokkos::parallel_reduce(
      "p3a::details::kokkos_simd_transform_reduce(1D)",
      kokkos_policy(0, quotient + 1),
      functor(binary_op,
        transform_b(
          transform_a(init, binary_op, unary_op),
          *first,
          *last)),
      reducer(init, binary_op, result));
  return result;
}

template <
  class SimdAbi,
  class ExecutionSpace,
  class Integral,
  class T,
  class BinaryReductionOp,
  class UnaryTransformOp>
[[nodiscard]]
std::enable_if_t<std::is_integral_v<Integral>, T>
kokkos_simd_transform_reduce(
    p3a::counting_iterator3<Integral> first,
    p3a::counting_iterator3<Integral> last,
    T init,
    BinaryReductionOp binary_op,
    UnaryTransformOp unary_op)
{
  auto const extents = last.vector - first.vector;
  if (extents.volume() == 0) return init;
  using transform_a = simd_reduce_wrapper<T, BinaryReductionOp, UnaryTransformOp>;
  using transform_b = simd_3d_functor<T, SimdAbi, Integral, transform_a>;
  using transform_c = kokkos_3d_functor<Integral, transform_b>;
  using functor = kokkos_3d_reduce_functor<
    T, BinaryReductionOp, transform_c, Integral>;
  using reducer = kokkos_reducer<T, BinaryReductionOp>;
  using kokkos_policy =
    Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::IndexType<Integral>,
      Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>;
  T result = init;
  Integral constexpr width = Integral(p3a::simd_mask<T, SimdAbi>::size());
  Integral const quotient = extents.x() / width;
  Kokkos::parallel_reduce(
      "p3a::details::kokkos_simd_transform_reduce(3D)",
      kokkos_policy(
        {Integral(0), first.vector.y(), first.vector.z()},
        {Integral(quotient + 1), last.vector.y(), last.vector.z()}),
      functor(binary_op,
        transform_c(
          transform_b(
            transform_a(init, binary_op, unary_op),
            first.vector.x(),
            last.vector.x()))),
      reducer(init, binary_op, result));
  return result;
}

}

template <
  class ExecutionPolicy,
  class Iterator,
  class T,
  class BinaryReductionOp,
  class UnaryTransformOp>
[[nodiscard]] T transform_reduce(
    ExecutionPolicy policy,
    Iterator first, Iterator last,
    T init,
    BinaryReductionOp binary_op,
    UnaryTransformOp unary_op)
{
  return details::kokkos_transform_reduce<typename ExecutionPolicy::kokkos_execution_space>(
      first, last, init, binary_op, unary_op);
}

template <
  class ExecutionPolicy,
  class T,
  class BinaryReductionOp,
  class UnaryTransformOp>
[[nodiscard]] T transform_reduce(
    ExecutionPolicy policy,
    subgrid3 subgrid,
    T init,
    BinaryReductionOp binary_op,
    UnaryTransformOp unary_op)
{
  return transform_reduce(
      policy,
      counting_iterator3<int>{subgrid.lower()},
      counting_iterator3<int>{subgrid.upper()},
      init,
      binary_op,
      unary_op);
}

template <
  class ExecutionPolicy,
  class Iterator,
  class T,
  class BinaryReductionOp,
  class UnaryTransformOp>
[[nodiscard]] T simd_transform_reduce(
    ExecutionPolicy policy,
    Iterator first, Iterator last,
    T init,
    BinaryReductionOp binary_op,
    UnaryTransformOp unary_op)
{
  return details::kokkos_simd_transform_reduce<
    typename ExecutionPolicy::simd_abi_type,
    typename ExecutionPolicy::kokkos_execution_space>(
      first, last, init, binary_op, unary_op);
}

template <
  class ExecutionPolicy,
  class T,
  class BinaryReductionOp,
  class UnaryTransformOp>
[[nodiscard]] T simd_transform_reduce(
    ExecutionPolicy policy,
    subgrid3 subgrid,
    T init,
    BinaryReductionOp binary_op,
    UnaryTransformOp unary_op)
{
  return simd_transform_reduce(
      policy,
      counting_iterator3<int>{subgrid.lower()},
      counting_iterator3<int>{subgrid.upper()},
      init,
      binary_op,
      unary_op);
}


namespace details {

template <
  class Allocator,
  class ExecutionPolicy>
class fixed_point_double_sum {
 public:
  using values_type = dynamic_array<double, Allocator, ExecutionPolicy>;
 private:
  mpicpp::comm m_comm;
  values_type m_values;
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

extern template class fixed_point_double_sum<device_allocator<double>, execution::parallel_policy>;

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
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE void operator()(SizeType i) const {
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
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE void operator()(vector3<int> const& grid_point) const {
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
  associative_sum<T, device_allocator<T>, execution::parallel_policy>;

}
