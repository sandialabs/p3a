#pragma once

#include <iterator>

#include "p3a_macros.hpp"

namespace p3a {

template <class Integral>
class counting_iterator {
  Integral m_value;
 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = Integral;
  using difference_type = std::make_signed_t<Integral>;
  using pointer = void*;
  using reference = value_type;
  P3A_ALWAYS_INLINE inline counting_iterator() = default;
  P3A_ALWAYS_INLINE inline constexpr counting_iterator(counting_iterator&&) = default;
  P3A_ALWAYS_INLINE inline constexpr counting_iterator& operator=(counting_iterator&&) = default;
  P3A_ALWAYS_INLINE inline constexpr counting_iterator(counting_iterator const&) = default;
  P3A_ALWAYS_INLINE inline constexpr counting_iterator& operator=(counting_iterator const&) = default;
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  explicit counting_iterator(Integral arg)
    :m_value(arg)
  {
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  bool operator==(counting_iterator const& other) const
  {
    return m_value == other.m_value;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  bool operator!=(counting_iterator const& other) const
  {
    return m_value != other.m_value;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  reference operator*() const
  {
    return m_value;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  pointer operator->() const
  {
    return nullptr;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  counting_iterator& operator++()
  {
    ++m_value;
    return *this;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  counting_iterator operator++(int)
  {
    auto result = *this;
    ++m_value;
    return result;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  counting_iterator& operator--()
  {
    --m_value;
    return *this;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  counting_iterator operator--(int)
  {
    auto result = *this;
    --m_value;
    return result;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  counting_iterator& operator+=(difference_type n)
  {
    m_value += n;
    return *this;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  counting_iterator& operator-=(difference_type n)
  {
    m_value -= n;
    return *this;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  counting_iterator operator+(difference_type n) const
  {
    return counting_iterator(m_value + n);
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  counting_iterator operator-(difference_type n) const
  {
    return counting_iterator(m_value - n);
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  difference_type operator-(counting_iterator const& other) const
  {
    return m_value - other.m_value;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  reference operator[](difference_type n) const
  {
    return m_value + n;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  bool operator<(counting_iterator const& other) const
  {
    return m_value < other.m_value;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  bool operator>(counting_iterator const& other) const
  {
    return m_value > other.m_value;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  bool operator<=(counting_iterator const& other) const
  {
    return m_value <= other.m_value;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  bool operator>=(counting_iterator const& other) const
  {
    return m_value >= other.m_value;
  }
};

template <class Integral>
P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
counting_iterator<Integral> operator+(
    typename counting_iterator<Integral>::difference_type n,
    counting_iterator<Integral> const& it)
{
  return it + n;
}

template <class Integral>
class counting_iterator3 {
 public:
  vector3<Integral> vector;
  P3A_ALWAYS_INLINE inline counting_iterator3() = default;
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline constexpr
  counting_iterator3(Integral a, Integral b, Integral c)
    :vector(a, b, c)
  {}
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline constexpr
  counting_iterator3(vector3<Integral> const& v)
    :vector(v)
  {}
};

}
