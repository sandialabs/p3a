#include "p3a_execution.hpp"

namespace p3a {

#ifdef __CUDACC__

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

void cuda_execution::synchronize() const {
  details::handle_cuda_error(
      cudaStreamSynchronize(stream));
}

#endif

#ifdef __HIPCC__

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

void hip_execution::synchronize() const {
  details::handle_hip_error(
      hipStreamSynchronize(stream));
}

#endif

}
