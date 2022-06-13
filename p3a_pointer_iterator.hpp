#pragma once

#include <cstddef>
#include <iterator>

#include "p3a_macros.hpp"
#include "p3a_simd.hpp"

namespace p3a {

template <class T>
class pointer_iterator {
 public:
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using pointer = value_type*;
  using iterator_category = std::random_access_iterator_tag;
 private:
  pointer m_pointer;
 public:
  P3A_ALWAYS_INLINE inline
  pointer_iterator() = default;
  P3A_ALWAYS_INLINE inline constexpr
  pointer_iterator(pointer_iterator&&) = default;
  P3A_ALWAYS_INLINE inline constexpr
  pointer_iterator(pointer_iterator const&) = default;
  P3A_ALWAYS_INLINE inline constexpr
  pointer_iterator& operator=(pointer_iterator&&) = default;
  P3A_ALWAYS_INLINE inline constexpr
  pointer_iterator& operator=(pointer_iterator const&) = default;
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  explicit pointer_iterator(pointer p)
    :m_pointer(p)
  {
  }
  // convert from non-const to const iterator
  template <class U,
    typename std::enable_if<
      std::is_same_v<U const, T> && (!std::is_const_v<U>),
      bool>::type = false>
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  pointer_iterator(pointer_iterator<U> const& other)
    :m_pointer(static_cast<U*>(other))
  {
  }
  // operators for EqualityComparable
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  bool operator==(pointer_iterator const& other) const
  {
    return m_pointer == other.m_pointer;
  }
  // operators for LegacyInputIterator
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  bool operator!=(pointer_iterator const& other) const
  {
    return m_pointer != other.m_pointer;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  reference operator*() const
  {
    return *m_pointer;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  pointer operator->() const
  {
    return m_pointer;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline
  pointer_iterator& operator++()
  {
    ++m_pointer;
    return *this;
  }
  // operators for LegacyForwardIterator
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline
  pointer_iterator operator++(int)
  {
    auto const temp = *this;
    ++m_pointer;
    return temp;
  }
  // operators for LegacyBidirectionalIterator
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline
  pointer_iterator& operator--()
  {
    --m_pointer;
    return *this;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline
  pointer_iterator operator--(int)
  {
    auto const temp = *this;
    --m_pointer;
    return temp;
  }
  // operators for LegacyRandomAccessIterator
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline
  pointer_iterator& operator+=(difference_type n)
  {
    m_pointer += n;
    return *this;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  pointer_iterator operator+(difference_type n) const
  {
    return pointer_iterator(m_pointer + n);
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline
  pointer_iterator& operator-=(difference_type n)
  {
    m_pointer += n;
    return *this;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  pointer_iterator operator-(difference_type n) const
  {
    return pointer_iterator(m_pointer + n);
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  reference operator[](difference_type n) const
  {
    return m_pointer[n];
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  bool operator<(pointer_iterator const& other) const
  {
    return m_pointer < other.m_pointer;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  bool operator>(pointer_iterator const& other) const
  {
    return m_pointer > other.m_pointer;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  bool operator>=(pointer_iterator const& other) const
  {
    return m_pointer >= other.m_pointer;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline constexpr
  bool operator<=(pointer_iterator const& other) const
  {
    return m_pointer <= other.m_pointer;
  }
};

template <class T, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
simd<T, Abi> load(
    pointer_iterator<T const> ptr, int offset, simd_mask<T, Abi> const& mask)
{
  simd<T, Abi> result;
  where(mask, result).copy_from(ptr + offset, element_aligned_tag());
  return result;
}

template <class T, class Integral, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
std::enable_if_t<std::is_integral_v<Integral>, simd<T, Abi>>
load(T const* ptr, simd<Integral, Abi> const& offset, simd_mask<T, Abi> const& mask)
{
  simd<T, Abi> result;
  where(mask, result).gather_from(ptr, offset);
  return result;
}

template <class T, class Abi>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
void store(
    simd<T, Abi> const& value,
    T* ptr,
    int offset,
    no_deduce_t<simd_mask<T, Abi>> const& mask)
{
  where(mask, value).copy_to(ptr + offset, element_aligned_tag());
}

template <class T, class Integral, class Abi>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
std::enable_if_t<std::is_integral_v<Integral>, void>
store(
    simd<T, Abi> const& value,
    T* ptr,
    simd<Integral, Abi> const& offset,
    no_deduce_t<simd_mask<T, Abi>> const& mask)
{
  where(mask, value).scatter_to(ptr, offset);
}

}
