#pragma once

#include "p3a_macros.hpp"

namespace p3a {

template <class T, int N>
class static_array {
  T m_members[N];
 public:
  using reference = T&;
  using const_reference = T const&;
  using size_type = int;
  using iterator = T*;
  using const_iterator = T const*;
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr
  reference operator[](size_type pos) { return m_members[pos]; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr
  const_reference operator[](size_type pos) const { return m_members[pos]; }
  [[nodiscard]] P3A_ALWAYS_INLINE static constexpr
  size_type size() { return N; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr
  T* data() { return m_members; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr
  T const* data() const { return m_members; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr
  iterator begin() { return data(); }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr
  const_iterator begin() const { return data(); }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr
  const_iterator cbegin() const { return data(); }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr
  iterator end() { return data() + size(); }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr
  const_iterator end() const { return data() + size(); }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr
  const_iterator cend() const { return data() + size(); }
};

}
