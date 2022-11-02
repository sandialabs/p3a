#pragma once

#include "p3a_macros.hpp"
#include "p3a_simd.hpp"

#include <exception>
#include <string>

#include <Kokkos_Core.hpp>

#ifdef KOKKOS_ENABLE_HIP
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-attributes"
#include <hip/hip_runtime.h>
#pragma clang diagnostic pop
#endif

#include <Kokkos_Core.hpp>

namespace p3a {

namespace execution {

class sequenced_policy {
 public:
  P3A_ALWAYS_INLINE constexpr void synchronize() const {}
  using simd_abi_type = simd_abi::scalar;
};

inline constexpr sequenced_policy seq = {};

class hot_policy {
 public:
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline constexpr
  void synchronize() const {}
  using simd_abi_type = simd_abi::scalar;
};

inline constexpr hot_policy hot = {};

class kokkos_serial_policy {
 public:
  void synchronize() const {}
  using simd_abi_type = simd_abi::ForSpace<Kokkos::DefaultHostExecutionSpace>;
  using kokkos_execution_space = Kokkos::Serial;
};

inline constexpr kokkos_serial_policy kokkos_serial = {};

}

#ifdef KOKKOS_ENABLE_CUDA

class cuda_exception : public std::exception
{
  std::string error_string;
 public:
  cuda_exception(cudaError_t error);
  virtual const char* what() const noexcept override;
};

namespace details {

void handle_cuda_error(cudaError_t error);

}

namespace execution {

class cuda_policy {
 public:
  void synchronize() const;
  using simd_abi_type = simd_abi::scalar;
  using kokkos_execution_space = Kokkos::Cuda;
};

inline constexpr cuda_policy cuda = {};

}

#endif

#ifdef KOKKOS_ENABLE_HIP

class hip_exception : public std::exception
{
  std::string error_string;
 public:
  hip_exception(hipError_t error);
  virtual const char* what() const noexcept override;
};

namespace details {

void handle_hip_error(hipError_t error);

}

namespace execution {

class hip_policy {
 public:
  void synchronize() const;
  using simd_abi_type = simd_abi::scalar;
  using kokkos_execution_space = Kokkos::HIP;
};

inline constexpr hip_policy hip = {};

}

#endif

#ifdef KOKKOS_ENABLE_OPENMP

namespace execution {

class openmp_policy {
 public:
  void synchronize() const {}
  using simd_abi_type = simd_abi::ForSpace<Kokkos::DefaultHostExecutionSpace>;
  using kokkos_execution_space = Kokkos::OpenMP;
};

inline constexpr openmp_policy openmp = {};

}

#endif

namespace execution {

#if defined(KOKKOS_ENABLE_CUDA)
using parallel_policy = cuda_policy;
#elif defined(KOKKOS_ENABLE_HIP)
using parallel_policy = hip_policy;
#elif defined(KOKKOS_ENABLE_OPENMP)
using parallel_policy = openmp_policy;
#else
using parallel_policy = kokkos_serial_policy;
#endif
inline constexpr parallel_policy par = {};

}

}
