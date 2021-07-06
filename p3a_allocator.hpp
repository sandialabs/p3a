#pragma once

#include <cstdint> //int64_t
#include <cstdlib> //malloc

namespace p3a {

template <class T>
class allocator {
 public:
  using size_type = std::int64_t;
  template <class U> struct rebind { using other = p3a::allocator<U>; };
  P3A_NEVER_INLINE static T* allocate(size_type n)
  {
    return static_cast<T*>(std::malloc(std::size_t(n) * sizeof(T)));
  }
  P3A_NEVER_INLINE static void deallocate(T* p, size_type)
  {
    std::free(p);
  }
};

#ifdef __CUDACC__

template <class T>
class cuda_host_allocator {
 public:
  using size_type = std::int64_t;
  template <class U> struct rebind { using other = p3a::cuda_host_allocator<U>; };
  P3A_NEVER_INLINE static T* allocate(size_type n)
  {
    void* ptr = nullptr;
    cudaMallocHost(&ptr, std::size_t(n) * sizeof(T));
    return static_cast<T*>(ptr);
  }
  P3A_NEVER_INLINE static void deallocate(T* p, size_type)
  {
    cudaFreeHost(p);
  }
};

template <class T>
class cuda_device_allocator {
 public:
  using size_type = std::int64_t;
  template <class U> struct rebind { using other = p3a::cuda_device_allocator<U>; };
  P3A_NEVER_INLINE static T* allocate(size_type n)
  {
    void* ptr = nullptr;
    cudaMalloc(&ptr, std::size_t(n) * sizeof(T));
    return static_cast<T*>(ptr);
  }
  P3A_NEVER_INLINE static void deallocate(T* p, size_type)
  {
    cudaFree(p);
  }
};

#endif

#ifdef __HIPCC__

template <class T>
class hip_host_allocator {
 public:
  using size_type = std::int64_t;
  template <class U> struct rebind { using other = p3a::cuda_host_allocator<U>; };
  P3A_NEVER_INLINE static T* allocate(size_type n)
  {
    void* ptr = nullptr;
    hipHostMalloc(&ptr, std::size_t(n) * sizeof(T), hipHostMallocDefault);
    return static_cast<T*>(ptr);
  }
  P3A_NEVER_INLINE static void deallocate(T* p, size_type)
  {
    hipHostFree(p);
  }
};

template <class T>
class hip_device_allocator {
 public:
  using size_type = std::int64_t;
  template <class U> struct rebind { using other = p3a::cuda_device_allocator<U>; };
  P3A_NEVER_INLINE static T* allocate(size_type n)
  {
    void* ptr = nullptr;
    hipMalloc(&ptr, std::size_t(n) * sizeof(T));
    return static_cast<T*>(ptr);
  }
  P3A_NEVER_INLINE static void deallocate(T* p, size_type)
  {
    hipFree(p);
  }
};

#endif

#if defined(__CUDACC__)
template <class T>
using device_allocator = cuda_device_allocator<T>;
#elif defined(__HIPCC__)
template <class T>
using device_allocator = hip_device_allocator<T>;
#else
template <class T>
using device_allocator = allocator<T>;
#endif

#if defined(__CUDACC__)
template <class T>
using mirror_allocator = cuda_host_allocator<T>;
#elif defined(__HIPCC__)
template <class T>
using mirror_allocator = hip_host_allocator<T>;
#else
template <class T>
using mirror_allocator = allocator<T>;
#endif

}
