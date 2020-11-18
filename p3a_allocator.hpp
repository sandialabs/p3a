#pragma once

#include <cstdint> //int64_t
#include <cstdlib> //malloc

namespace p3a {

class host_allocator {
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
    std::free(p);
  }
};

#if defined(__CUDACC__)

class cuda_allocator {
 public:
  using size_type = std::int64_t;
  template <class T>
  CPL_NEVER_INLINE static T* allocate(size_type n)
  {
    void* ptr = nullptr;
    cudaMalloc(&ptr, std::size_t(n) * sizeof(T));
    return static_cast<T*>(ptr);
  }
  template <class T>
  CPL_NEVER_INLINE static void deallocate(T* p, size_type)
  {
    cudaFree(p);
  }
};

#endif

}
