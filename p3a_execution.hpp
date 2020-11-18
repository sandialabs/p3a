#pragma once

#include "p3a_macros.hpp"

namespace p3a {

class serial_execution {
};

inline constexpr serial_execution serial = {};

class local_execution {
};

inline constexpr local_execution local = {};

#ifdef __CUDACC__

class cuda_execution {
};

inline constexpr cuda_execution cuda = {};

#endif

#ifdef __CUDACC__
using device_execution = cuda_execution;
#else
using device_execution = serial_execution;
#endif
inline constexpr device_execution device = {};

}
