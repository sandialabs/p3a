#pragma once

#include "p3a_functions.hpp"
#include "p3a_mpi.hpp"
#include "p3a_allocator.hpp"
#include "p3a_execution.hpp"
#include "p3a_dynamic_array.hpp"
#include "p3a_reduce.hpp"

extern "C" void p3a_mpi_int128_sum(
    void* a,
    void* b,
    int*,
    MPI_Datatype*);

namespace p3a {

[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
int double_exponent(std::uint64_t as_int)
{
  auto const biased_exponent = (as_int >> 52) & 0b11111111111ull;
  return int(biased_exponent) - 1023;
}

[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
int exponent(double x)
{
  auto const as_int = p3a::bit_cast<std::uint64_t>(x);
  return double_exponent(as_int);
}

[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
std::uint64_t double_mantissa(std::uint64_t as_int)
{
  return as_int & 0b1111111111111111111111111111111111111111111111111111ull;
}

[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
std::uint64_t mantissa(double x)
{
  auto const as_int = p3a::bit_cast<std::uint64_t>(x);
  return double_mantissa(as_int);
}

[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
int double_sign_bit(std::uint64_t as_int)
{
  return int((as_int & 0b1000000000000000000000000000000000000000000000000000000000000000ull) >> 63);
}

[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
int sign_bit(double x)
{
  auto const as_int = p3a::bit_cast<std::uint64_t>(x);
  return double_sign_bit(as_int);
}

P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
void decompose_double(double value, int& sign_bit, int& exponent, std::uint64_t& mantissa)
{
  std::uint64_t const as_int = p3a::bit_cast<std::uint64_t>(value);
  sign_bit = p3a::double_sign_bit(as_int);
  exponent = p3a::double_exponent(as_int);
  mantissa = p3a::double_mantissa(as_int);
}

[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
double compose_double(int sign_bit_arg, int exponent_arg, std::uint64_t mantissa_arg)
{
  std::uint64_t const as_int = mantissa_arg |
      (std::uint64_t(exponent_arg + 1023) << 52) |
      (std::uint64_t(sign_bit_arg) << 63);
  return p3a::bit_cast<double>(as_int);
}

// value = significand * (2 ^ exponent)
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
void decompose_double(double value, std::int64_t& significand, int& exponent)
{
  int sign_bit;
  std::uint64_t mantissa;
  decompose_double(value, sign_bit, exponent, mantissa);
  if (exponent > -1023) {
    mantissa |= 0b10000000000000000000000000000000000000000000000000000ull;
  }
  significand = mantissa;
  if (sign_bit) significand = -significand;
  exponent -= 52;
}

[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
double compose_double(std::int64_t significand, int exponent)
{
  int sign_bit;
  if (significand < 0) {
    sign_bit = 1;
    significand = -significand;
  } else {
    sign_bit = 0;
  }
  auto constexpr maximum_significand =
    0b11111111111111111111111111111111111111111111111111111ull;
  while (significand > maximum_significand) {
    significand >>= 1;
    ++exponent;
  }
  auto constexpr minimum_significand =
    0b10000000000000000000000000000000000000000000000000000ull;
  while (significand < minimum_significand) {
    significand <<= 1;
    --exponent;
  }
  exponent += 52;
  // subnormals
  while (exponent < -1023) {
    significand >>= 1;
    ++exponent;
  }
  // infinity
  if (exponent > 1023) {
    significand = 0;
    exponent = 1023;
  }
  std::uint64_t mantissa =
    std::uint64_t(significand) & 0b1111111111111111111111111111111111111111111111111111ull;
  return compose_double(sign_bit, exponent, mantissa);
}

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
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static inline constexpr
  int128 from_double(double value, double unit) {
    return int128(std::int64_t(value / unit));
  }
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
  std::int64_t high() const { return m_high; }
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
  std::uint64_t low() const { return m_low; }
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline
  double to_double(double unit) const;
};

P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
int128 operator+(int128 const& a, int128 const& b) {
  auto high = a.high() + b.high();
  auto const low = a.low() + b.low();
  // check for overflow of low 64 bits, add carry to high
  high += (low < a.low());
  return int128(high, low);
}

P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
int128 operator-(int128 const& a, int128 const& b) {
  auto high = a.high() - b.high();
  auto const low = a.low() - b.low();
  // check for underflow of low 64 bits, subtract carry from high
  high -= (low > a.low());
  return int128(high, low);
}

P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
int128 operator-(int128 const& x) {
  return int128(0) - x;
}

P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
int128 operator>>(int128 const& x, int expo) {
  auto const low =
    (x.low() >> expo) |
    (std::uint64_t(x.high()) << (64 - expo));
  auto const high = x.high() >> expo;
  return int128(high, low);
}

P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
bool operator==(int128 const& lhs, int128 const& rhs) {
  return lhs.high() == rhs.high() && lhs.low() == rhs.low();
}

P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
bool operator<(int128 const& lhs, int128 const& rhs) {
  if (lhs.high() != rhs.high()) {
    return lhs.high() < rhs.high();
  }
  return lhs.low() < rhs.low();
}

P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline
double int128::to_double(double unit) const {
  int128 tmp = *this;
  if (tmp < int128(0)) tmp = -tmp;
  while (tmp.high()) {
    tmp = tmp >> 1;
    unit *= 2;
  }
  double x = tmp.low();
  if (*this < int128(0)) x = -x;
  x *= unit;
  return x;
}

template <
  class T,
  class Allocator = allocator<double>,
  class ExecutionPolicy = serial_execution>
class fixed_point_sum;

/* A reproducible sum of floating-point values.
   this operation is one of the key places where
   a program's output begins to depend on parallel
   partitioning and traversal order, because
   floating-point values do not produce the same
   sum when added in a different order.

   IEEE 754 64-bit floating point format is assumed,
   which has 52 bits in the fraction.

   The idea here is to add the numbers as fixed-point values.
   max_exponent() finds the largest exponent (e) such that
   all values are (<= 2^(e)).
   We then use the value (2^(e - 52)) as the unit, and sum all
   values as integers in that unit.
   This is guaranteed to be at least as accurate as the
   worst-case ordering of the values, i.e. being added
   in order of decreasing magnitude.

   If we used a 64-bit integer type, we would only be
   able to reliably add up to (2^12 = 4096) values
   (64 - 52 = 12).
   Thus we use a 128-bit integer type.
   This allows us to reliably add up to (2^76 > 10^22) values.
   By comparison, supercomputers today
   support a maximum of one million MPI ranks (10^6)
   and each rank typically can't hold more than
   one billion values (10^9), for a total of (10^15) values.
*/

template <
  class Allocator,
  class ExecutionPolicy>
class fixed_point_sum<double, Allocator, ExecutionPolicy> {
  mpi::comm m_comm;
  dynamic_array<double, Allocator, ExecutionPolicy> m_values;
  reducer<int, ExecutionPolicy> m_exponent_reducer;
  reducer<int128, ExecutionPolicy> m_int128_reducer;
 public:
  fixed_point_sum() = default;
  explicit fixed_point_sum(
      mpi::comm&& comm_arg)
    :m_comm(std::move(comm_arg))
  {}
  fixed_point_sum(fixed_point_sum&&) = default;
  fixed_point_sum& operator=(fixed_point_sum&&) = default;
  fixed_point_sum(fixed_point_sum const&) = delete;
  fixed_point_sum& operator=(fixed_point_sum const&) = delete;
#ifdef __CUDACC__
 public:
#else
 private:
#endif
  [[nodiscard]] P3A_NEVER_INLINE
  double reduce_stored_values()
  {
    int constexpr minimum_exponent =
      std::numeric_limits<int>::lowest();
    int const local_max_exponent =
      m_exponent_reducer.transform_reduce(
          m_values.cbegin(), m_values.cend(),
          minimum_exponent,
          maximizes<int>,
    [=] P3A_HOST P3A_DEVICE (double const& value) P3A_ALWAYS_INLINE {
      if (value == 0.0) return minimum_exponent;
      int exponent;
      std::frexp(value, &exponent);
      return exponent;
    });
    int global_max_exponent = local_max_exponent;
    m_comm.iallreduce(
        &global_max_exponent, 1, mpi::op::max());
    if (global_max_exponent == minimum_exponent) return 0.0;
    int constexpr mantissa_bits = 52;
    double const unit = std::exp2(
        double(global_max_exponent - mantissa_bits));
    int128 const local_sum =
      m_int128_reducer.transform_reduce(
          m_values.cbegin(), m_values.cend(),
          int128(0),
          adds<int128>,
    [=] P3A_HOST P3A_DEVICE (double const& value) P3A_ALWAYS_INLINE {
      return int128::from_double(value, unit);
    });
    int128 global_sum = local_sum;
    auto const int128_mpi_sum_op = 
      mpi::op::create(p3a_mpi_int128_sum);
    m_comm.iallreduce(
        MPI_IN_PLACE,
        &global_sum,
        sizeof(int128),
        mpi::datatype::predefined_packed(),
        int128_mpi_sum_op);
    return global_sum.to_double(unit);
  }
 public:
  template <class Iterator, class UnaryOp>
  [[nodiscard]] P3A_NEVER_INLINE
  double transform_reduce(
      Iterator first,
      Iterator last,
      UnaryOp unary_op)
  {
    auto const n = (last - first);
    m_values.resize(n);
    auto const policy = m_values.get_execution_policy();
    auto const values = m_values.begin();
    using size_type = std::remove_const_t<decltype(n)>;
    for_each(policy,
        counting_iterator<size_type>(0),
        counting_iterator<size_type>(n),
    [=] P3A_HOST P3A_DEVICE (size_type i) P3A_ALWAYS_INLINE {
      values[i] = unary_op(first[i]);
    });
    return reduce_stored_values();
  }
  template <class UnaryOp>
  [[nodiscard]] P3A_NEVER_INLINE
  double transform_reduce(
      subgrid3 grid,
      UnaryOp unary_op)
  {
    m_values.resize(grid.size());
    auto const policy = m_values.get_execution_policy();
    auto const values = m_values.begin();
    for_each(policy, grid,
    [=] P3A_HOST P3A_DEVICE (vector3<int> const& grid_point) P3A_ALWAYS_INLINE {
      int const index = grid.index(grid_point);
      values[index] = unary_op(grid_point);
    });
    return reduce_stored_values();
  }
};

template <class T>
using device_fixed_point_sum = 
  fixed_point_sum<T, device_allocator<T>, device_execution>;
template <class T>
using host_fixed_point_sum = 
  fixed_point_sum<
    T, allocator<T>, serial_execution>;

}
