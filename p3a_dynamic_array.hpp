#pragma once

#include "p3a_memory.hpp"
#include "p3a_allocator.hpp"

namespace p3a {

template <
  class T,
  class Allocator = allocator<T>,
  class ExecutionPolicy = serial_execution>
class dynamic_array {
 public:
  using size_type = std::int64_t;
  using iterator = T*;
  using const_iterator = T const*;
  using allocator_type = Allocator;
  using execution_policy = ExecutionPolicy;
  using value_type = T;
 private:
  T* m_begin;
  size_type m_size;
  size_type m_capacity;
  allocator_type m_allocator;
  execution_policy m_execution_policy;
 public:
  P3A_NEVER_INLINE dynamic_array()
   :m_begin(nullptr)
   ,m_size(0)
   ,m_capacity(0)
  {}
  P3A_NEVER_INLINE ~dynamic_array()
  {
    if (m_begin != nullptr) {
      destroy(m_execution_policy, m_begin, m_begin + m_size);
      m_allocator.deallocate(m_begin, m_capacity);
      m_begin = nullptr;
      m_size = 0;
      m_capacity = 0;
    }
  }
  P3A_NEVER_INLINE dynamic_array(dynamic_array&& other)
    :m_begin(other.m_begin)
    ,m_size(other.m_size)
    ,m_capacity(other.m_capacity)
    ,m_execution_policy(std::move(other.m_execution_policy))
  {
    other.m_begin = nullptr;
    other.m_size = 0;
    other.m_capacity = 0;
  }
  P3A_NEVER_INLINE dynamic_array& operator=(dynamic_array&& other)
  {
    m_begin = other.m_begin;
    m_size = other.m_size;
    m_capacity = other.m_capacity;
    m_execution_policy = std::move(other.m_execution_policy);
    other.m_begin = nullptr;
    other.m_size = 0;
    other.m_capacity = 0;
    return *this;
  }
  dynamic_array(dynamic_array const& other)
    :m_begin(nullptr)
    ,m_size(0)
    ,m_capacity(0)
    ,m_execution_policy(other.m_execution_policy)
  {
    reserve(other.capacity());
    m_size = other.size();
    uninitialized_copy(m_execution_policy, other.begin(), other.end(), m_begin);
  }
  dynamic_array& operator=(dynamic_array const& other)
  {
    resize(0);
    reserve(other.capacity());
    m_size = other.size();
    uninitialized_copy(m_execution_policy, other.begin(), other.end(), m_begin);
    return *this;
  }
  explicit dynamic_array(size_type size_in)
    :dynamic_array()
  {
    resize(size_in);
  }
 private:
  P3A_NEVER_INLINE void increase_capacity(size_type new_capacity)
  {
    T* const new_allocation = m_allocator.allocate(new_capacity);
    uninitialized_move(m_execution_policy, m_begin, m_begin + m_size, new_allocation);
    destroy(m_execution_policy, m_begin, m_begin + m_size);
    m_allocator.deallocate(m_begin, m_capacity);
    m_begin = new_allocation;
    m_capacity = new_capacity;
  }
 public:
  P3A_NEVER_INLINE void reserve(size_type const count)
  {
    if (count <= m_capacity) return;
    size_type const new_capacity = maximum(count, 2 * m_capacity);
    increase_capacity(new_capacity);
  }
  P3A_NEVER_INLINE void resize(size_type const count)
  {
    if (m_size == count) return;
    reserve(count);
    size_type const common_size = minimum(m_size, count);
    destroy(m_execution_policy, m_begin + common_size, m_begin + m_size);
    uninitialized_default_construct(m_execution_policy, m_begin + common_size, m_begin + count);
    m_size = count;
  }
  P3A_NEVER_INLINE void resize(size_type const count, value_type const& value)
  {
    if (m_size == count) return;
    reserve(count);
    size_type const common_size = minimum(m_size, count);
    destroy(m_execution_policy, m_begin + common_size, m_begin + m_size);
    uninitialized_fill(m_execution_policy, m_begin + common_size, m_begin + count, value);
    m_size = count;
  }
  void push_back(T&& value) {
    reserve(m_size + 1);
    ::new (static_cast<void*>(m_begin + m_size)) T(std::move(value));
    ++m_size;
  }
  void push_back(T const& value) {
    reserve(m_size + 1);
    ::new (static_cast<void*>(m_begin + m_size)) T(value);
    ++m_size;
  }
  T& back()
  {
    return operator[](size() - 1);
  }
  T const& back() const
  {
    return operator[](size() - 1);
  }
  void pop_back()
  {
    resize(size() - 1);
  }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr T* data() { return m_begin; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr T const* data() const { return m_begin; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr iterator begin() { return m_begin; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr const_iterator begin() const { return m_begin; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr const_iterator cbegin() const { return m_begin; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr iterator end() { return m_begin + m_size; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr const_iterator end() const { return m_begin + m_size; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr const_iterator cend() const { return m_begin + m_size; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr size_type size() const { return m_size; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr size_type capacity() const { return m_capacity; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr bool empty() const { return m_size == 0; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr T& operator[](size_type pos) { return m_begin[pos]; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr T const& operator[](size_type pos) const { return m_begin[pos]; }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr
  allocator_type get_allocator() const {
    return m_allocator;
  }
  [[nodiscard]] P3A_ALWAYS_INLINE constexpr
  execution_policy get_execution_policy() const {
    return m_execution_policy;
  }
};

#ifdef __CUDACC__
template <class T>
using device_array = dynamic_array<T, cuda_device_allocator<T>, cuda_execution>;
template <class T>
using mirror_array = dynamic_array<T, cuda_host_allocator<T>, serial_execution>;
template <class T>
using host_array = dynamic_array<T, allocator<T>, serial_execution>;
#else
template <class T>
using device_array = dynamic_array<T, allocator<T>, serial_execution>;
template <class T>
using mirror_array = dynamic_array<T, allocator<T>, serial_execution>;
template <class T>
using host_array = dynamic_array<T, allocator<T>, serial_execution>;
#endif

}
