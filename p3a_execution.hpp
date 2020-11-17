#pragma once

#include <cstdint> //int64_t
#include <cstdlib> //malloc

#include "p3a_macros.hpp"

namespace p3a {

namespace execution {

class serial_policy {
 public:
  using size_type = std::int64_t;
  template <class T>
  CPL_NEVER_INLINE static T* allocate(size_type n)
  {
    return static_cast<T*>(std::malloc(std::size_t(n) * sizeof(T)));
  }
  template <class T>
  CPL_NEVER_INLINE static void deallocate(T* p, size_type)
  {
    return std::free(p);
  }
};

inline constexpr serial_policy serial = {};

class local_policy {
};

inline constexpr local_policy local = {};

}

}
