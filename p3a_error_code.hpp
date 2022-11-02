#pragma once

#include <type_traits>

#include <Kokkos_Core.hpp>

#include "p3a_dynamic_array.hpp"

namespace p3a {

class device_error_code {
  int* m_pointer;
 public:
  device_error_code(int* pointer_arg)
    :m_pointer(pointer_arg)
  {
  }
  device_error_code(device_error_code&&) = default;
  device_error_code(device_error_code const&) = default;
  device_error_code& operator=(device_error_code&&) = default;
  device_error_code& operator=(device_error_code const&) = default;
  template <class T,
           std::enable_if_t<std::is_enum_v<T> || std::is_integral_v<T>, bool> = false>
  P3A_HOST_DEVICE void operator=(T const& rhs) const
  {
    Kokkos::atomic_store(m_pointer, static_cast<int>(rhs));
  }
};

class error_code {
  host_pinned_array<int> m_array;
 public:
  error_code()
  {
    m_array.resize(1, 0);
  }
  device_error_code on_device()
  {
    return device_error_code(m_array.data());
  }
  template <class T>
  T value() const
  {
    p3a::execution::par.synchronize();
    return static_cast<T>(m_array[0]);
  }
};

}
