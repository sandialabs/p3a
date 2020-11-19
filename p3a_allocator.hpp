#pragma once

#include <cstdint> //int64_t
#include <cstdlib> //malloc

namespace p3a {

template <class T>
class allocator {
 public:
  using size_type = std::int64_t;
  CPL_NEVER_INLINE static T* allocate(size_type n)
  {
    return static_cast<T*>(std::malloc(std::size_t(n) * sizeof(T)));
  }
  CPL_NEVER_INLINE static void deallocate(T* p, size_type)
  {
    std::free(p);
  }
};

#if defined(__CUDACC__)

template <class T>
class cuda_host_allocator {
 public:
  using size_type = std::int64_t;
  CPL_NEVER_INLINE static T* allocate(size_type n)
  {
    void* ptr = nullptr;
    cudaMallocHost(&ptr, std::size_t(n) * sizeof(T));
    return static_cast<T*>(ptr);
  }
  CPL_NEVER_INLINE static void deallocate(T* p, size_type)
  {
    cudaFreeHost(p);
  }
};

template <class T>
class cuda_device_allocator {
 public:
  using size_type = std::int64_t;
  CPL_NEVER_INLINE static T* allocate(size_type n)
  {
    void* ptr = nullptr;
    cudaMalloc(&ptr, std::size_t(n) * sizeof(T));
    return static_cast<T*>(ptr);
  }
  CPL_NEVER_INLINE static void deallocate(T* p, size_type)
  {
    cudaFree(p);
  }
};

#endif

#ifdef __CUDACC__
template <class T>
using device_allocator = cuda_device_allocator<T>;
#else
template <class T>
using device_allocator = allocator<T>;
#endif

#ifdef __CUDACC__
template <class T>
using mirror_allocator = cuda_host_allocator<T>;
#else
template <class T>
using mirror_allocator = allocator<T>;
#endif

}
