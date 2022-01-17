#pragma once

#include "p3a_reduce.hpp"

namespace p3a {

namespace details {

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
  sign_bit = double_sign_bit(as_int);
  exponent = double_exponent(as_int);
  mantissa = double_mantissa(as_int);
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
  if (significand != 0) {
    while (significand < minimum_significand) {
      significand <<= 1;
      --exponent;
    }
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
    exponent = 1024;
  }
  std::uint64_t mantissa =
    std::uint64_t(significand) & 0b1111111111111111111111111111111111111111111111111111ull;
  return compose_double(sign_bit, exponent, mantissa);
}

[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
std::int64_t fixed_point_right_shift(std::int64_t significand, int shift)
{
  int sign;
  std::uint64_t unsigned_significand;
  if (significand < 0) {
    sign = -1;
    unsigned_significand = -significand;
  } else {
    sign = 1;
    unsigned_significand = significand;
  }
  if (shift > 64) {
    unsigned_significand = 0;
  } else {
    unsigned_significand >>= shift;
  }
  significand = sign * unsigned_significand;
  return significand;
}

[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
std::int64_t decompose_double(double value, int maximum_exponent)
{
  int exponent;
  std::int64_t significand;
  decompose_double(value, significand, exponent);
  printf("value %.17e = %lld * (2 ^ %d)\n", value, significand, exponent);
  return fixed_point_right_shift(significand, maximum_exponent - exponent);
}

[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
int128 operator+(int128 const& a, int128 const& b) {
  auto high = a.high() + b.high();
  auto const low = a.low() + b.low();
  // check for overflow of low 64 bits, add carry to high
  high += (low < a.low());
  return int128(high, low);
}

P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline
void operator+=(int128& a, int128 const& b)
{
  a = a + b;
}

[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
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

[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
int128 operator>>(int128 const& x, int expo) {
  auto const low =
    (x.low() >> expo) |
    (std::uint64_t(x.high()) << (64 - expo));
  auto const high = x.high() >> expo;
  return int128(high, low);
}

P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline
void operator>>=(int128& x, int expo)
{
  x = x >> expo;
}

P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
bool operator==(int128 const& lhs, int128 const& rhs) {
  return lhs.high() == rhs.high() && lhs.low() == rhs.low();
}

[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
bool operator<(int128 const& lhs, int128 const& rhs) {
  if (lhs.high() != rhs.high()) {
    return lhs.high() < rhs.high();
  }
  return lhs.low() < rhs.low();
}

[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
bool operator>(int128 const& lhs, int128 const& rhs) {
  return rhs < lhs;
}

[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline
double compose_double(int128 significand_128, int exponent)
{
  int sign;
  if (significand_128 < int128(0)) {
    sign = -1;
    significand_128 = -significand_128;
  } else {
    sign = 1;
  }
  int128 const maximum_significand_128(
    0b11111111111111111111111111111111111111111111111111111ll);
  while (significand_128 > maximum_significand_128) {
    significand_128 >>= 1;
    ++exponent;
  }
  std::int64_t const significand_64 = sign * p3a::bit_cast<std::int64_t>(significand_128.low());
  return compose_double(significand_64, exponent);
}

}

}
