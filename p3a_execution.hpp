#pragma once

#include "p3a_macros.hpp"

namespace p3a {

class serial_execution {
};

inline constexpr serial_execution serial = {};

class serial_local_execution {
};

inline constexpr serial_local_execution serial_local = {};

#ifdef __CUDACC__

class cuda_execution {
};

inline constexpr cuda_execution cuda = {};

class cuda_local_execution {
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
