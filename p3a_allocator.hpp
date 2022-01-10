#pragma once

#include <cstdint> //int64_t
#include <cstdlib> //malloc

namespace p3a {

class allocation_failure : public std::bad_alloc {
  char message[100];
  std::int64_t n;
 public:
  allocation_failure(char const* memory_space_arg, std::int64_t n_arg)
  {
    using long_long = long long;
    auto const long_long_n = long_long(n_arg);
    std::snprintf(message, sizeof(message), "failed to allocate %lld bytes in %s memory",
        long_long_n, memory_space_arg);
  }
  virtual const char* what() const noexcept override
  {
    return message;
  }
};

template <class T>
class allocator {
 public:
  using size_type = std::int64_t;
  template <class U> struct rebind { using other = p3a::allocator<U>; };
  P3A_NEVER_INLINE static T* allocate(size_type n)
  {
    auto const result = std::malloc(std::size_t(n) * sizeof(T));
    if ((result == nullptr) && (n != 0)) {
      throw allocation_failure("CPU", n);
    }
    return static_cast<T*>(result);
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
    auto const result = cudaMallocHost(&ptr, std::size_t(n) * sizeof(T));
    if (result != cudaSuccess) {
      throw allocation_failure("CUDA host pinned", n);
    }
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
    auto const result = cudaMalloc(&ptr, std::size_t(n) * sizeof(T));
    if (result != cudaSuccess) {
      throw allocation_failure("CUDA device", n);
    }
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
  template <class U> struct rebind { using other = p3a::hip_host_allocator<U>; };
  P3A_NEVER_INLINE static T* allocate(size_type n)
  {
    void* ptr = nullptr;
    auto const result = hipHostMalloc(&ptr, std::size_t(n) * sizeof(T), hipHostMallocDefault);
    if (result != hipSuccess) {
      throw allocation_failure("HIP host pinned", n);
    }
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
  template <class U> struct rebind { using other = p3a::hip_device_allocator<U>; };
  P3A_NEVER_INLINE static T* allocate(size_type n)
  {
    void* ptr = nullptr;
    auto const result = hipMalloc(&ptr, std::size_t(n) * sizeof(T));
    if (result != hipSuccess) {
      throw allocation_failure("HIP device", n);
    }
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
