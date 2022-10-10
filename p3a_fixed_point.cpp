#include "p3a_fixed_point.hpp"

extern "C" void p3a_mpi_int128_sum(
    void* a,
    void* b,
    int*,
    MPI_Datatype*)
{
  p3a::details::int128* a2 = static_cast<p3a::details::int128*>(a);
  p3a::details::int128* b2 = static_cast<p3a::details::int128*>(b);
  *b2 = *b2 + *a2;
}

namespace p3a {

namespace details {

/* A fixed-point sum of floating-point values.
   this operation is one of the key places where
   a program's output begins to depend on parallel
   partitioning and traversal order, because
   floating-point addition is not associative.

   IEEE 754 64-bit floating point format is assumed,
   which has 52 bits in the mantissa.

   The idea here is to add the numbers as fixed-point values.
   First, find the largest exponent of all the floating-point values.
   Then, convert all floating-point numbers to fixed-point at
   the largest exponent found.
   Finally, add up these fixed-point numbers (integers),
   which is an associative sum.
   This is guaranteed to be at least as accurate as the
   worst-case ordering of the values, i.e. being added
   in order of decreasing magnitude.

   If we used a 64-bit integer type, we would only be
   able to reliably add up to (2^(64 - 53) = 2^11 = 2048) values.
   This is why we use a custom 128-bit integer type.
   This allows us to reliably add up to (2^(128-53) = 2^75 > 10^22) values.
   By comparison, supercomputers today
   support a maximum of one million MPI ranks (10^6)
   and each rank typically can't hold more than
   one billion values (10^9), for a total of (10^15) values.
*/

template <
  class Allocator,
  class ExecutionPolicy>
[[nodiscard]] P3A_NEVER_INLINE
double fixed_point_double_sum<Allocator, ExecutionPolicy>::compute()
{
  int constexpr minimum_exponent = -1075;
  int const value_count = int(m_values.size());
  auto const values = m_values.cbegin();
  using simd_abi_type = typename ExecutionPolicy::simd_abi_type;
  int const local_max_exponent =
  simd_transform_reduce(
        ExecutionPolicy(),
        counting_iterator<int>(0),
        counting_iterator<int>(value_count),
        minimum_exponent,
        maximizes<int>,
  [=] P3A_HOST_DEVICE (int i, simd_mask<std::int32_t, simd_abi_type> const& mask) P3A_ALWAYS_INLINE {
    auto const value = load(values, i, simd_mask<double, simd_abi_type>(mask));
    simd<std::int64_t, simd_abi_type> significand;
    simd<std::int32_t, simd_abi_type> exponent;
    decompose_double(value, significand, exponent);
    return exponent;
  });
  int global_max_exponent = local_max_exponent;
  m_comm.iallreduce(
      &global_max_exponent, 1, mpicpp::op::max());
  int128 const local_sum =
  transform_reduce(
        ExecutionPolicy(),
        counting_iterator<int>(0),
        counting_iterator<int>(value_count),
        int128(0),
        adds<int128>,
  [=] P3A_HOST_DEVICE (int i) P3A_ALWAYS_INLINE {
    auto const value = load(values, i);
    auto significand_64 = decompose_double(value, global_max_exponent);
    return significand_64;
  });
  int128 global_sum = local_sum;
  auto const int128_mpi_sum_op = 
    mpicpp::op::create(p3a_mpi_int128_sum);
  m_comm.iallreduce(
      MPI_IN_PLACE,
      &global_sum,
      sizeof(int128),
      mpicpp::datatype::predefined_packed(),
      int128_mpi_sum_op);
  return compose_double(global_sum, global_max_exponent);
}

// explicitly instantiate for host and device so the actual reduction
// code can stay in this translation unit
//
template class fixed_point_double_sum<device_allocator<double>, execution::parallel_policy>;

}

}
