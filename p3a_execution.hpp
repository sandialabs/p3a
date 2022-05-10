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

class host_execution {
 public:
  P3A_ALWAYS_INLINE constexpr void synchronize() const {}
  using simd_abi_type = simd_abi::scalar;
};

inline constexpr host_execution host = {};

class host_device_execution {
 public:
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline constexpr
  void synchronize() const {}
  using simd_abi_type = simd_abi::scalar;
};

inline constexpr host_device_execution host_device = {};

class kokkos_serial_execution {
 public:
  void synchronize() const {}
  using simd_abi_type = simd_abi::host_native;
  using kokkos_execution_space = Kokkos::Serial;
};

inline constexpr kokkos_serial_execution kokkos_serial = {};

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

class cuda_execution {
 cudaStream_t stream{nullptr};
 public:
  void synchronize() const;
  using simd_abi_type = simd_abi::scalar;
  using kokkos_execution_space = Kokkos::Cuda;
};

inline constexpr cuda_execution cuda = {};

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

class hip_execution {
  hipStream_t stream{nullptr};
 public:
  void synchronize() const;
  using simd_abi_type = simd_abi::scalar;
  using kokkos_execution_space = Kokkos::Hip;
};

inline constexpr hip_execution hip = {};

#endif

#if defined(KOKKOS_ENABLE_CUDA)
using device_execution = cuda_execution;
#elif defined(KOKKOS_ENABLE_HIP)
using device_execution = hip_execution;
#else
using device_execution = kokkos_serial_execution;
#endif
inline constexpr device_execution device = {};

}
