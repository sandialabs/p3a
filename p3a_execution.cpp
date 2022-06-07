#include "p3a_execution.hpp"

namespace p3a {

#ifdef KOKKOS_ENABLE_CUDA

cuda_exception::cuda_exception(cudaError_t error)
  :error_string(cudaGetErrorString(error))
{
}

const char* cuda_exception::what() const noexcept
{
  return error_string.c_str();
}

namespace details {

void handle_cuda_error(cudaError_t error)
{
  if (error == cudaSuccess) return;
  throw cuda_exception(error);
}

}

namespace execution {

void cuda_policy::synchronize() const {
  details::handle_cuda_error(
      cudaStreamSynchronize(nullptr));
}

}

#endif

#ifdef KOKKOS_ENABLE_HIP

hip_exception::hip_exception(hipError_t error)
  :error_string(hipGetErrorString(error))
{
}

const char* hip_exception::what() const noexcept
{
  return error_string.c_str();
}

namespace details {

void handle_hip_error(hipError_t error)
{
  if (error == hipSuccess) return;
  throw hip_exception(error);
}

}

namespace execution {

void hip_policy::synchronize() const {
  details::handle_hip_error(
      hipStreamSynchronize(nullptr));
}

}

#endif

}
