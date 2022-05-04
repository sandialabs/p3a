#pragma once

#include "p3a_macros.hpp"
#include "p3a_simd.hpp"

#include <exception>
#include <string>

#ifdef __HIPCC__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-attributes"
#include <hip/hip_runtime.h>
#pragma clang diagnostic pop
#endif

#include <Kokkos_Core.hpp>

namespace p3a {

class serial_execution {
 public:
  void synchronize() const {}
  using simd_abi_type = simd_abi::host_native;
  using kokkos_execution_space = Kokkos::Serial;
};

inline constexpr serial_execution serial = {};

class serial_local_execution {
 public:
  P3A_ALWAYS_INLINE constexpr void synchronize() const {}
  using simd_abi_type = simd_abi::scalar;
};

inline constexpr serial_local_execution serial_local = {};

class local_execution {
 public:
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  void synchronize() const {}
  using simd_abi_type = simd_abi::scalar;
};

inline constexpr local_execution local = {};

#ifdef __CUDACC__

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

class cuda_local_execution {
 public:
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE constexpr void synchronize() const
  {
  }
  using simd_abi_type = simd_abi::scalar;
};

inline constexpr cuda_local_execution cuda_local = {};

#endif

#ifdef __HIPCC__

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

class hip_local_execution {
 public:
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE constexpr void synchronize() const
  {
  }
  using simd_abi_type = simd_abi::scalar;
};

inline constexpr hip_local_execution hip_local = {};

#endif

#if defined(__CUDACC__)
using device_execution = cuda_execution;
using device_local_execution = cuda_local_execution;
#elif defined(__HIPCC__)
using device_execution = hip_execution;
using device_local_execution = hip_local_execution;
#else
using device_execution = serial_execution;
using device_local_execution = serial_local_execution;
#endif
using host_execution = serial_execution;
inline constexpr host_execution host = {};
inline constexpr device_execution device = {};
inline constexpr device_local_execution device_local = {};

}
