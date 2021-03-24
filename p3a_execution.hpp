#pragma once

#include "p3a_macros.hpp"

#include <exception>
#include <string>

namespace p3a {

class serial_execution {
 public:
  void synchronize() const {}
};

inline constexpr serial_execution serial = {};

class serial_local_execution {
 public:
  P3A_ALWAYS_INLINE constexpr void synchronize() const {}
};

inline constexpr serial_local_execution serial_local = {};

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
};

inline constexpr cuda_execution cuda = {};

class cuda_local_execution {
 public:
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE constexpr void synchronize() const
  {
  }
};

inline constexpr cuda_local_execution cuda_local = {};

#endif

#ifdef __CUDACC__
using device_execution = cuda_execution;
using device_local_execution = cuda_local_execution;
#else
using device_execution = serial_execution;
using device_local_execution = serial_local_execution;
#endif
inline constexpr device_execution device = {};
inline constexpr device_local_execution device_local = {};

}
