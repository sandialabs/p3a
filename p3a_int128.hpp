#pragma once

#include <cstdint>

#include "p3a_macros.hpp"

namespace p3a {

class int128 {
  std::int64_t m_high;
  std::uint64_t m_low;
 public:
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
  double to_double(double unit) const {
    int128 tmp = *this;
    if (tmp < int128(0)) tmp = -tmp;
    while (tmp.high()) {
      tmp = tmp >> 1;
      unit *= 2;
    }
    double x = tmp.low;
    if (*this < int128(0)) x = -x;
    x *= unit;
    return x;
  }
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

}
