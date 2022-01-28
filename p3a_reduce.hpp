#pragma once

#include "mpicpp.hpp"

#include "p3a_quantity.hpp"
#include "p3a_execution.hpp"
#include "p3a_grid3.hpp"
#include "p3a_dynamic_array.hpp"
#include "p3a_counting_iterator.hpp"
#include "p3a_functional.hpp"

namespace p3a {

template <class T, class ExecutionPolicy>
class reducer;

template <class T>
class reducer<T, serial_execution> {
  serial_execution m_policy;
 public:
  reducer() = default;
  reducer(reducer&&) = default;
  reducer& operator=(reducer&&) = default;
  reducer(reducer const&) = delete;
  reducer& operator=(reducer const&) = delete;
  template <class Iterator, class BinaryOp, class UnaryOp>
  [[nodiscard]] P3A_NEVER_INLINE
  T transform_reduce(
      Iterator first,
      Iterator last,
      T init,
      BinaryOp binary_op,
      UnaryOp unary_op)
  {
    for (; first != last; ++first) {
      init = binary_op(init, unary_op(*first));
    }
    return init;
  }
  template <class BinaryOp, class UnaryOp>
  [[nodiscard]] P3A_NEVER_INLINE
  T transform_reduce(
      subgrid3 grid,
      T init,
      BinaryOp binary_op,
      UnaryOp unary_op)
  {
    for_each(m_policy, grid,
    [&] (vector3<int> const& item) P3A_ALWAYS_INLINE {
      init = binary_op(init, unary_op(item));
    });
    return init;
  }
  template <class BinaryOp, class UnaryOp>
  [[nodiscard]] P3A_NEVER_INLINE
  T simd_transform_reduce(
      subgrid3 grid,
      T init,
      BinaryOp binary_op,
      UnaryOp unary_op)
  {
    T const identity_value = init;
    simd_for_each<T>(m_policy, grid,
    [&] (vector3<int> const& item, host_simd_mask<T> const& mask) P3A_ALWAYS_INLINE {
      simd<T, simd_abi::host_native> const simd_value = unary_op(item, mask);
      T const scalar_value = reduce(where(mask, simd_value), identity_value, binary_op);
      init = binary_op(init, scalar_value);
    });
    return init;
  }
};

template <
    class T,
    class BinaryOp,
    class UnaryOp>
[[nodiscard]] P3A_NEVER_INLINE
T transform_reduce(
    serial_execution policy,
    subgrid3 subgrid,
    T init,
    BinaryOp binary_op,
    UnaryOp unary_op)
{
  reducer<T, serial_execution> r;
  return r.transform_reduce(subgrid, init, binary_op, unary_op);
}

#ifdef __CUDACC__

namespace details {

static constexpr int cuda_reducer_threads_per_block = 256;
static constexpr int cuda_grid_reducer_threads_per_block = 64;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct cuda_shared_memory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct cuda_shared_memory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

template <int Count, bool FitsInInt = (Count <= int(sizeof(int)))>
class cuda_recursive_sliced_shuffle_helper;

template <int Count>
class cuda_recursive_sliced_shuffle_helper<Count, true>
{
  int val;
 public:
  __device__ P3A_ALWAYS_INLINE void shuffle_down(unsigned int delta)
  {
    val = __shfl_down_sync(0xFFFFFFFF, val, delta, 32);
  }
};

template <int Count>
class cuda_recursive_sliced_shuffle_helper<Count, false>
{
  int val;
  cuda_recursive_sliced_shuffle_helper<Count - int(sizeof(int))> next;
 public:
  __device__ P3A_ALWAYS_INLINE void shuffle_down(unsigned int delta)
  {
    val = __shfl_down_sync(0xFFFFFFFF, val, delta, 32);
    next.shuffle_down(delta);
  }
};

template <class T>
__device__ P3A_ALWAYS_INLINE inline
T cuda_shuffle_down(T element, unsigned int delta)
{
  if constexpr (
      std::is_same_v<T, int> ||
      std::is_same_v<T, unsigned int> ||
      std::is_same_v<T, long> ||
      std::is_same_v<T, unsigned long> ||
      std::is_same_v<T, long long> ||
      std::is_same_v<T, unsigned long long> ||
      std::is_same_v<T, float> ||
      std::is_same_v<T, double>)
  {
    return __shfl_down_sync(0xFFFFFFFF, element, delta, 32);
  } else {
    cuda_recursive_sliced_shuffle_helper<sizeof(T)> helper;
    static_assert(std::is_trivially_copyable_v<T>, "reduction types need to be trivially copyable to shuffle in CUDA");
    memcpy(&helper, &element, sizeof(T));
    helper.shuffle_down(delta);
    memcpy(&element, &helper, sizeof(T));
    return element;
  }
  return element;
}

template <class ForwardIt, class T, class BinaryOp, class UnaryOp>
__global__ void cuda_reduce(ForwardIt first, T* g_odata, int n, T init, BinaryOp binop, UnaryOp unop) {
  constexpr int blockSize = cuda_reducer_threads_per_block;
  T* sdata = cuda_shared_memory<T>();
  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  int tid = threadIdx.x;
  int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
  int gridSize = (blockSize * 2) * gridDim.x;
  T myResult = init;
  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    myResult = binop(myResult, unop(first[i]));
    // ensure we don't read out of bounds
    if (i + blockSize < n) myResult = binop(myResult, unop(first[i + blockSize]));
    i += gridSize;
  }
  // each thread puts its local sum into shared memory
  sdata[tid] = myResult;
  __syncthreads();
  // do reduction in shared mem
  if (tid < 128) {
    sdata[tid] = myResult = binop(myResult, sdata[tid + 128]);
  }
  __syncthreads();
  if (tid < 64) {
    sdata[tid] = myResult = binop(myResult, sdata[tid + 64]);
  }
  __syncthreads();
  if (tid < 32) {
    // Fetch final intermediate sum from 2nd warp
    myResult = binop(myResult, sdata[tid + 32]);
    // Reduce final warp using shuffle
    for (unsigned int offset = 32 / 2; offset > 0; offset /= 2) {
      myResult = binop(myResult, cuda_shuffle_down(myResult, offset));
    }
  }
  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = myResult;
}

template <class T, class BinaryOp, class UnaryOp>
__global__ void cuda_grid_reduce(
    vector3<int> first, vector3<int> last,
    T* g_odata, T init, BinaryOp binop, UnaryOp unop) {
  constexpr int blockSize = cuda_grid_reducer_threads_per_block;
  T* sdata = cuda_shared_memory<T>();
  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  vector3<int> const thread_idx(threadIdx.x, threadIdx.y, threadIdx.z);
  vector3<int> const block_idx(blockIdx.x, blockIdx.y, blockIdx.z);
  grid3 const grid_dim(gridDim.x, gridDim.y, gridDim.z);
  vector3<int> const user_extents = last - first;
  int const thread_i = thread_idx.x();
  int const block_i = grid_dim.index(block_idx);
  int const tid = thread_i;
  int const x_i = thread_idx.x() + (block_idx.x() * blockSize);
  int const y_i = block_idx.y();
  int const z_i = block_idx.z();
  vector3<int> const xyz(x_i, y_i, z_i);
  T myResult = init;
  if (x_i < user_extents.x()) {
    myResult = binop(myResult, unop(xyz + first));
  }
  // each thread puts its local sum into shared memory
  sdata[tid] = myResult;
  __syncthreads();
  // do reduction in shared mem
  if (tid < 32) {
    // Fetch final intermediate sum from 2nd warp
    myResult = binop(myResult, sdata[tid + 32]);
    // Reduce final warp using shuffle
    for (unsigned int offset = 32 / 2; offset > 0; offset /= 2) {
      myResult = binop(myResult, cuda_shuffle_down(myResult, offset));
    }
  }
  // write result for this block to global mem
  if (tid == 0) g_odata[block_i] = myResult;
}

template <class T, class BinaryOp, class UnaryOp>
__global__ void cuda_simd_grid_reduce(
    vector3<int> first, vector3<int> last,
    T* g_odata, T init, BinaryOp binop, UnaryOp unop) {
  constexpr int blockSize = cuda_grid_reducer_threads_per_block;
  T* sdata = cuda_shared_memory<T>();
  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  vector3<int> const thread_idx(threadIdx.x, threadIdx.y, threadIdx.z);
  vector3<int> const block_idx(blockIdx.x, blockIdx.y, blockIdx.z);
  grid3 const grid_dim(gridDim.x, gridDim.y, gridDim.z);
  vector3<int> const user_extents = last - first;
  int const thread_i = thread_idx.x();
  int const block_i = grid_dim.index(block_idx);
  int const tid = thread_i;
  int const x_i = thread_idx.x() + (block_idx.x() * blockSize);
  int const y_i = block_idx.y();
  int const z_i = block_idx.z();
  vector3<int> const xyz(x_i, y_i, z_i);
  auto const mask = simd_mask<T, simd_abi::scalar>(x_i < user_extents.x());
  simd<T, simd_abi::scalar> const simd_value = unop(xyz + first, mask);
  T myResult = reduce(where(mask, simd_value), init, binop);
  // each thread puts its local sum into shared memory
  sdata[tid] = myResult;
  __syncthreads();
  // do reduction in shared mem
  if (tid < 32) {
    // Fetch final intermediate sum from 2nd warp
    myResult = binop(myResult, sdata[tid + 32]);
    // Reduce final warp using shuffle
    for (unsigned int offset = 32 / 2; offset > 0; offset /= 2) {
      myResult = binop(myResult, cuda_shuffle_down(myResult, offset));
    }
  }
  // write result for this block to global mem
  if (tid == 0) g_odata[block_i] = myResult;
}

}

template <class T>
class reducer<T, cuda_execution> {
  dynamic_array<
    T,
    cuda_device_allocator<T>,
    cuda_execution> m_storage;
  T* m_scratch1;
  T* m_scratch2;
  static constexpr int threads_per_block = details::cuda_reducer_threads_per_block;
  static constexpr int grid_threads_per_block = details::cuda_grid_reducer_threads_per_block;
  static vector3<int> get_block_grid(vector3<int> user_grid) {
    return vector3<int>(
        (user_grid.x() + grid_threads_per_block - 1) / grid_threads_per_block,
        user_grid.y(),
        user_grid.z());
  }
  static int get_num_grid_blocks(vector3<int> user_grid) {
    return get_block_grid(user_grid).volume();
  }
  static int get_num_blocks(int size) {
    return minimum(64, (size + threads_per_block - 1) / threads_per_block);
  }
  template <class ForwardIt, class BinaryOp, class UnaryOp>
  void reduction_pass(int size, int blocks, ForwardIt first, T* d_odata, T init, BinaryOp binop, UnaryOp unop) {
    dim3 const dimBlock(threads_per_block, 1, 1);
    dim3 const dimGrid(blocks, 1, 1);
    int const smemSize = threads_per_block * sizeof(T);
    cudaStream_t const cuda_stream = nullptr;
    details::cuda_reduce<<<
      dimGrid,
      dimBlock,
      smemSize,
      cuda_stream>>>(first, d_odata, size, init, binop, unop);
  }
  template <class BinaryOp, class UnaryOp>
  void grid_reduction_pass(
      vector3<int> first, vector3<int> last, vector3<int> block_grid,
      T* d_odata, T init, BinaryOp binop, UnaryOp unop) {
    dim3 const dimBlock(grid_threads_per_block, 1, 1);
    dim3 const dimGrid(block_grid.x(), block_grid.y(), block_grid.z());
    int const smemSize = grid_threads_per_block * sizeof(T);
    cudaStream_t const cuda_stream = nullptr;
    details::cuda_grid_reduce<<<
      dimGrid,
      dimBlock,
      smemSize,
      cuda_stream>>>(first, last, d_odata, init, binop, unop);
  }
  template <class BinaryOp, class UnaryOp>
  void simd_grid_reduction_pass(
      vector3<int> first, vector3<int> last, vector3<int> block_grid,
      T* d_odata, T init, BinaryOp binop, UnaryOp unop) {
    dim3 const dimBlock(grid_threads_per_block, 1, 1);
    dim3 const dimGrid(block_grid.x(), block_grid.y(), block_grid.z());
    int const smemSize = grid_threads_per_block * sizeof(T);
    cudaStream_t const cuda_stream = nullptr;
    details::cuda_simd_grid_reduce<<<
      dimGrid,
      dimBlock,
      smemSize,
      cuda_stream>>>(first, last, d_odata, init, binop, unop);
  }
  template <class ForwardIt, class BinaryOp, class UnaryOp>
  T* reduce_on_device(
      int n, ForwardIt first, T init, BinaryOp binop, UnaryOp unop) {
    int blocks = get_num_blocks(n);
    this->reduction_pass(n, blocks, first, m_scratch2, init, binop, unop);
    int s = blocks;
    T* tmp_d_idata = m_scratch1;
    T* tmp_d_odata = m_scratch2;
    while (s > 1) {
      std::swap(tmp_d_idata, tmp_d_odata);
      blocks = get_num_blocks(s);
      this->reduction_pass(s, blocks, tmp_d_idata, tmp_d_odata, init, binop, identity<T>());
      s = (s + (threads_per_block * 2 - 1)) / (threads_per_block * 2);
    }
    return tmp_d_odata;
  }
  template <class BinaryOp, class UnaryOp>
  T* grid_reduce_on_device(vector3<int> first, vector3<int> last, T init, BinaryOp binop, UnaryOp unop) {
    int blocks = get_num_grid_blocks(last - first);
    vector3<int> const block_grid = get_block_grid(last - first);
    this->grid_reduction_pass(first, last, block_grid, m_scratch2, init, binop, unop);
    int s = blocks;
    T* tmp_d_idata = m_scratch1;
    T* tmp_d_odata = m_scratch2;
    while (s > 1) {
      std::swap(tmp_d_idata, tmp_d_odata);
      blocks = get_num_blocks(s);
      this->reduction_pass(s, blocks, tmp_d_idata, tmp_d_odata, init, binop, identity<T>());
      s = (s + (threads_per_block * 2 - 1)) / (threads_per_block * 2);
    }
    return tmp_d_odata;
  }
  template <class BinaryOp, class UnaryOp>
  T* simd_grid_reduce_on_device(vector3<int> first, vector3<int> last, T init, BinaryOp binop, UnaryOp unop) {
    int blocks = get_num_grid_blocks(last - first);
    vector3<int> const block_grid = get_block_grid(last - first);
    this->simd_grid_reduction_pass(first, last, block_grid, m_scratch2, init, binop, unop);
    int s = blocks;
    T* tmp_d_idata = m_scratch1;
    T* tmp_d_odata = m_scratch2;
    while (s > 1) {
      std::swap(tmp_d_idata, tmp_d_odata);
      blocks = get_num_blocks(s);
      this->reduction_pass(s, blocks, tmp_d_idata, tmp_d_odata, init, binop, identity<T>());
      s = (s + (threads_per_block * 2 - 1)) / (threads_per_block * 2);
    }
    return tmp_d_odata;
  }
  template <class ForwardIt, class BinaryOp, class UnaryOp>
  T reduce_to_host(int n, ForwardIt first, T init, BinaryOp binop, UnaryOp unop) {
    m_storage.resize(2 * n);
    m_scratch1 = m_storage.data();
    m_scratch2 = m_storage.data() + n;
    T host_result = init;
    T const* device_result_ptr = this->reduce_on_device(n, first, init, binop, unop);
    cudaMemcpy(&host_result, device_result_ptr, sizeof(T), cudaMemcpyDefault);
    m_storage.resize(0);
    m_scratch1 = nullptr;
    m_scratch2 = nullptr;
    return host_result;
  }
  template <class BinaryOp, class UnaryOp>
  T grid_reduce_to_host(vector3<int> first, vector3<int> last, T init, BinaryOp binop, UnaryOp unop) {
    auto const n = (last - first).volume();
    m_storage.resize(2 * n);
    m_scratch1 = m_storage.data();
    m_scratch2 = m_storage.data() + n;
    T host_result = init;
    T const* device_result_ptr = this->grid_reduce_on_device(first, last, init, binop, unop);
    cudaMemcpy(&host_result, device_result_ptr, sizeof(T), cudaMemcpyDefault);
    m_storage.resize(0);
    m_scratch1 = nullptr;
    m_scratch2 = nullptr;
    return host_result;
  }
  template <class BinaryOp, class UnaryOp>
  T simd_grid_reduce_to_host(vector3<int> first, vector3<int> last, T init, BinaryOp binop, UnaryOp unop) {
    auto const n = (last - first).volume();
    m_storage.resize(2 * n);
    m_scratch1 = m_storage.data();
    m_scratch2 = m_storage.data() + n;
    T host_result = init;
    T const* device_result_ptr = this->simd_grid_reduce_on_device(first, last, init, binop, unop);
    cudaMemcpy(&host_result, device_result_ptr, sizeof(T), cudaMemcpyDefault);
    m_storage.resize(0);
    m_scratch1 = nullptr;
    m_scratch2 = nullptr;
    return host_result;
  }
 public:
  reducer()
    :m_scratch1(nullptr)
    ,m_scratch2(nullptr)
  {}
  reducer(reducer&&) = default;
  reducer& operator=(reducer&&) = default;
  reducer(reducer const&) = delete;
  reducer& operator=(reducer const&) = delete;
  template <class ForwardIt, class BinaryOp, class UnaryOp>
  [[nodiscard]]
  T transform_reduce(
      ForwardIt first, ForwardIt last,
      T init, BinaryOp binop, UnaryOp unop) {
    return this->reduce_to_host(last - first, first, init, binop, unop);
  }
  template <class BinaryOp, class UnaryOp>
  [[nodiscard]]
  T transform_reduce(
      subgrid3 grid,
      T init, BinaryOp binop, UnaryOp unop) {
    return this->grid_reduce_to_host(grid.lower(), grid.upper(), init, binop, unop);
  }
  template <class BinaryOp, class UnaryOp>
  [[nodiscard]]
  T simd_transform_reduce(
      subgrid3 grid,
      T init, BinaryOp binop, UnaryOp unop) {
    return this->simd_grid_reduce_to_host(grid.lower(), grid.upper(), init, binop, unop);
  }
};

template <
    class T,
    class BinaryOp,
    class UnaryOp>
[[nodiscard]] P3A_NEVER_INLINE
T transform_reduce(
    cuda_execution policy,
    subgrid3 subgrid,
    T init,
    BinaryOp binary_op,
    UnaryOp unary_op)
{
  reducer<T, cuda_execution> r;
  return r.transform_reduce(subgrid, init, binary_op, unary_op);
}

#endif

#ifdef __HIPCC__

namespace details {

static constexpr int hip_reducer_threads_per_block = 256;
static constexpr int hip_grid_reducer_threads_per_block = 128;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct hip_shared_memory {
  __device__ inline operator T *() {
    HIP_DYNAMIC_SHARED(int, __smem);
    return (T *)__smem;
  }
  __device__ inline operator const T *() const {
    HIP_DYNAMIC_SHARED(int, __smem);
    return (T *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct hip_shared_memory<double> {
  __device__ inline operator double *() {
    HIP_DYNAMIC_SHARED(double, __smem_d);
    return (double *)__smem_d;
  }
  __device__ inline operator const double *() const {
    HIP_DYNAMIC_SHARED(double, __smem_d);
    return (double *)__smem_d;
  }
};

template <int Count, bool FitsInInt = (Count <= int(sizeof(int)))>
class hip_recursive_sliced_shuffle_helper;

template <int Count>
class hip_recursive_sliced_shuffle_helper<Count, true>
{
  int val;
 public:
  __device__ P3A_ALWAYS_INLINE void shuffle_down(unsigned int delta)
  {
    val = __shfl_down(val, delta, 64);
  }
};

template <int Count>
class hip_recursive_sliced_shuffle_helper<Count, false>
{
  int val;
  hip_recursive_sliced_shuffle_helper<Count - int(sizeof(int))> next;
 public:
  __device__ P3A_ALWAYS_INLINE void shuffle_down(unsigned int delta)
  {
    val = __shfl_down(val, delta, 64);
    next.shuffle_down(delta);
  }
};

template <class T>
__device__ P3A_ALWAYS_INLINE T hip_shuffle_down(T val, unsigned int delta)
{
  if constexpr (
      std::is_same_v<T, int> ||
      std::is_same_v<T, unsigned int> ||
      std::is_same_v<T, float> ||
      std::is_same_v<T, double> ||
      std::is_same_v<T, long> ||
      std::is_same_v<T, long long>)
  {
    return __shfl_down(val, delta, 64);
  } else {
    hip_recursive_sliced_shuffle_helper<int(sizeof(T))> helper;
    memcpy(&helper, &val, sizeof(T));
    helper.shuffle_down(delta);
    memcpy(&val, &helper, sizeof(T));
    return val;
  }
}

template <class ForwardIt, class T, class BinaryOp, class UnaryOp>
__global__ void hip_reduce(ForwardIt first, T* g_odata, int n, T init, BinaryOp binop, UnaryOp unop) {
  constexpr int blockSize = hip_reducer_threads_per_block;
  // Handle to thread block group
  T* sdata = hip_shared_memory<T>();
  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  int tid = hipThreadIdx_x;
  int i = hipBlockIdx_x * (blockSize * 2) + hipThreadIdx_x;
  int gridSize = (blockSize * 2) * hipGridDim_x;
  T myResult = init;
  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    myResult = binop(myResult, unop(first[i]));
    // ensure we don't read out of bounds
    if (i + blockSize < n) myResult = binop(myResult, unop(first[i + blockSize]));
    i += gridSize;
  }
  // each thread puts its local sum into shared memory
  sdata[tid] = myResult;
  __syncthreads();
  // do reduction in shared mem
  if (tid < 128) {
    sdata[tid] = myResult = binop(myResult, sdata[tid + 128]);
  }
  __syncthreads();
  if (tid < 64) {
    // Fetch final intermediate sum from 2nd warp
    myResult = binop(myResult, sdata[tid + 64]);
    // Reduce final warp using shuffle
    for (unsigned int offset = 64 / 2; offset > 0; offset /= 2) {
      myResult = binop(myResult, hip_shuffle_down(myResult, offset));
    }
  }
  // write result for this block to global mem
  if (tid == 0) g_odata[hipBlockIdx_x] = myResult;
}

template <class T, class BinaryOp, class UnaryOp>
__global__ void hip_grid_reduce(
    vector3<int> first, vector3<int> last,
    T* g_odata, T init, BinaryOp binop, UnaryOp unop) {
  constexpr int blockSize = hip_grid_reducer_threads_per_block;
  // Handle to thread block group
  T* sdata = hip_shared_memory<T>();
  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  vector3<int> const thread_idx(hipThreadIdx_x, hipThreadIdx_y, hipThreadIdx_z);
  vector3<int> const block_idx(hipBlockIdx_x, hipBlockIdx_y, hipBlockIdx_z);
  grid3 const grid_dim(hipGridDim_x, hipGridDim_y, hipGridDim_z);
  vector3<int> const user_extents = last - first;
  int const thread_i = thread_idx.x();
  int const block_i = grid_dim.index(block_idx);
  int const tid = thread_i;
  int const x_i = thread_idx.x() + (block_idx.x() * blockSize);
  int const y_i = block_idx.y();
  int const z_i = block_idx.z();
  vector3<int> const xyz(x_i, y_i, z_i);
  T myResult = init;
  if (x_i < user_extents.x()) {
    myResult = binop(myResult, unop(xyz + first));
  }
  // each thread puts its local sum into shared memory
  sdata[tid] = myResult;
  __syncthreads();
  // do reduction in shared mem
  if (tid < 64) {
    // Fetch final intermediate sum from 2nd warp
    myResult = binop(myResult, sdata[tid + 64]);
    // Reduce final warp using shuffle
    for (unsigned int offset = 64 / 2; offset > 0; offset /= 2) {
      myResult = binop(myResult, hip_shuffle_down(myResult, offset));
    }
  }
  // write result for this block to global mem
  if (tid == 0) g_odata[block_i] = myResult;
}

}

template <class T>
class reducer<T, hip_execution> {
  dynamic_array<
    T,
    hip_device_allocator<T>,
    hip_execution> m_storage;
  T* m_scratch1;
  T* m_scratch2;
  static constexpr int threads_per_block = details::hip_reducer_threads_per_block;
  static constexpr int grid_threads_per_block = details::hip_grid_reducer_threads_per_block;
  static vector3<int> get_block_grid(vector3<int> user_grid) {
    return vector3<int>(
        (user_grid.x() + grid_threads_per_block - 1) / grid_threads_per_block,
        user_grid.y(),
        user_grid.z());
  }
  static int get_num_grid_blocks(vector3<int> user_grid) {
    return get_block_grid(user_grid).volume();
  }
  static int get_num_blocks(int size) {
    return (size + threads_per_block - 1) / threads_per_block;
  }
  template <class ForwardIt, class BinaryOp, class UnaryOp>
  void reduction_pass(int size, int blocks, ForwardIt first, T* d_odata, T init, BinaryOp binop, UnaryOp unop) {
    dim3 const dimBlock(threads_per_block, 1, 1);
    dim3 const dimGrid(blocks, 1, 1);
    int const smemSize = threads_per_block * sizeof(T);
    hipStream_t const hip_stream = nullptr;
    hipLaunchKernelGGL(
      details::hip_reduce,
      dimGrid,
      dimBlock,
      smemSize,
      hip_stream,
      first,
      d_odata,
      size,
      init,
      binop,
      unop);
  }
  template <class BinaryOp, class UnaryOp>
  void grid_reduction_pass(
      vector3<int> first, vector3<int> last, vector3<int> block_grid,
      T* d_odata, T init, BinaryOp binop, UnaryOp unop) {
    dim3 const dimBlock(grid_threads_per_block, 1, 1);
    dim3 const dimGrid(block_grid.x(), block_grid.y(), block_grid.z());
    int const smemSize = grid_threads_per_block * sizeof(T);
    hipStream_t const hip_stream = nullptr;
    hipLaunchKernelGGL(
      details::hip_grid_reduce,
      dimGrid,
      dimBlock,
      smemSize,
      hip_stream,
      first,
      last,
      d_odata,
      init,
      binop,
      unop);
  }
  template <class ForwardIt, class BinaryOp, class UnaryOp>
  T* reduce_on_device(
      int n, ForwardIt first, T init, BinaryOp binop, UnaryOp unop) {
    int blocks = get_num_blocks(n);
    this->reduction_pass(n, blocks, first, m_scratch2, init, binop, unop);
    int s = blocks;
    T* tmp_d_idata = m_scratch1;
    T* tmp_d_odata = m_scratch2;
    while (s > 1) {
      std::swap(tmp_d_idata, tmp_d_odata);
      blocks = get_num_blocks(s);
      this->reduction_pass(s, blocks, tmp_d_idata, tmp_d_odata, init, binop, identity<T>());
      s = (s + (threads_per_block * 2 - 1)) / (threads_per_block * 2);
    }
    return tmp_d_odata;
  }
  template <class BinaryOp, class UnaryOp>
  T* grid_reduce_on_device(vector3<int> first, vector3<int> last, T init, BinaryOp binop, UnaryOp unop) {
    int blocks = get_num_grid_blocks(last - first);
    vector3<int> const block_grid = get_block_grid(last - first);
    this->grid_reduction_pass(first, last, block_grid, m_scratch2, init, binop, unop);
    int s = blocks;
    T* tmp_d_idata = m_scratch1;
    T* tmp_d_odata = m_scratch2;
    while (s > 1) {
      std::swap(tmp_d_idata, tmp_d_odata);
      blocks = get_num_blocks(s);
      this->reduction_pass(s, blocks, tmp_d_idata, tmp_d_odata, init, binop, identity<T>());
      s = (s + (threads_per_block * 2 - 1)) / (threads_per_block * 2);
    }
    return tmp_d_odata;
  }
  template <class ForwardIt, class BinaryOp, class UnaryOp>
  T reduce_to_host(int n, ForwardIt first, T init, BinaryOp binop, UnaryOp unop) {
    m_storage.resize(2 * n);
    m_scratch1 = m_storage.data();
    m_scratch2 = m_storage.data() + n;
    T host_result = init;
    T const* device_result_ptr = this->reduce_on_device(n, first, init, binop, unop);
    hipMemcpy(&host_result, device_result_ptr, sizeof(T), hipMemcpyDefault);
    m_storage.resize(0);
    m_scratch1 = nullptr;
    m_scratch2 = nullptr;
    return host_result;
  }
  template <class BinaryOp, class UnaryOp>
  T grid_reduce_to_host(vector3<int> first, vector3<int> last, T init, BinaryOp binop, UnaryOp unop) {
    auto const n = (last - first).volume();
    m_storage.resize(2 * n);
    m_scratch1 = m_storage.data();
    m_scratch2 = m_storage.data() + n;
    T host_result = init;
    T const* device_result_ptr = this->grid_reduce_on_device(first, last, init, binop, unop);
    hipMemcpy(&host_result, device_result_ptr, sizeof(T), hipMemcpyDefault);
    m_storage.resize(0);
    m_scratch1 = nullptr;
    m_scratch2 = nullptr;
    return host_result;
  }
 public:
  reducer()
    :m_scratch1(nullptr)
    ,m_scratch2(nullptr)
  {}
  reducer(reducer&&) = default;
  reducer& operator=(reducer&&) = default;
  reducer(reducer const&) = delete;
  reducer& operator=(reducer const&) = delete;
  template <class BinaryOp, class UnaryOp>
  [[nodiscard]]
  T transform_reduce(
      subgrid3 grid,
      T init, BinaryOp binop, UnaryOp unop) {
    return this->grid_reduce_to_host(grid.lower(), grid.upper(), init, binop, unop);
  }
  template <class ForwardIt, class BinaryOp, class UnaryOp>
  [[nodiscard]]
  T transform_reduce(
      ForwardIt first, ForwardIt last,
      T init, BinaryOp binop, UnaryOp unop) {
    return this->reduce_to_host(last - first, first, init, binop, unop);
  }
};

template <
    class T,
    class BinaryOp,
    class UnaryOp>
[[nodiscard]] P3A_NEVER_INLINE
T transform_reduce(
    hip_execution policy,
    subgrid3 subgrid,
    T init,
    BinaryOp binary_op,
    UnaryOp unary_op)
{
  reducer<T, hip_execution> r;
  return r.transform_reduce(subgrid, init, binary_op, unary_op);
}

#endif

namespace details {

class int128 {
  std::int64_t m_high;
  std::uint64_t m_low;
 public:
  P3A_ALWAYS_INLINE inline int128() = default;
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
  int128(std::int64_t high_arg, std::uint64_t low_arg)
    :m_high(high_arg)
    ,m_low(low_arg)
  {}
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
  int128(std::int64_t value)
    :int128(
        std::int64_t(-1) * (value < 0),
        std::uint64_t(value))
  {}
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
  std::int64_t high() const { return m_high; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
  std::uint64_t low() const { return m_low; }
};

template <
  class Allocator,
  class ExecutionPolicy>
class fixed_point_double_sum {
 public:
  using values_type = dynamic_array<double, Allocator, ExecutionPolicy>;
 private:
  mpicpp::comm m_comm;
  values_type m_values;
  reducer<int, ExecutionPolicy> m_exponent_reducer;
  reducer<int128, ExecutionPolicy> m_int128_reducer;
 public:
  fixed_point_double_sum() = default;
  explicit fixed_point_double_sum(mpicpp::comm&& comm_arg)
    :m_comm(std::move(comm_arg))
  {}
  fixed_point_double_sum(fixed_point_double_sum&&) = default;
  fixed_point_double_sum& operator=(fixed_point_double_sum&&) = default;
  fixed_point_double_sum(fixed_point_double_sum const&) = delete;
  fixed_point_double_sum& operator=(fixed_point_double_sum const&) = delete;
 public:
  [[nodiscard]] P3A_NEVER_INLINE
  double compute();
  [[nodiscard]] P3A_ALWAYS_INLINE inline constexpr
  values_type& values() { return m_values; }
};

extern template class fixed_point_double_sum<allocator<double>, serial_execution>;
#ifdef __CUDACC__
extern template class fixed_point_double_sum<cuda_device_allocator<double>, cuda_execution>;
#endif
#ifdef __HIPCC__
extern template class fixed_point_double_sum<hip_device_allocator<double>, hip_execution>;
#endif

}

template <class T, class Allocator, class ExecutionPolicy>
class associative_sum;

template <
  class Allocator,
  class ExecutionPolicy>
class associative_sum<double, Allocator, ExecutionPolicy> {
  details::fixed_point_double_sum<Allocator, ExecutionPolicy> m_fixed_point;
 public:
  associative_sum() = default;
  explicit associative_sum(mpicpp::comm&& comm_arg)
    :m_fixed_point(std::move(comm_arg))
  {}
  associative_sum(associative_sum&&) = default;
  associative_sum& operator=(associative_sum&&) = default;
  associative_sum(associative_sum const&) = delete;
  associative_sum& operator=(associative_sum const&) = delete;
#ifdef __CUDACC__
 public:
#else
 private:
#endif
 public:
  template <class Iterator, class UnaryOp>
  [[nodiscard]] P3A_NEVER_INLINE
  double transform_reduce(
      Iterator first,
      Iterator last,
      UnaryOp unary_op)
  {
    auto const n = (last - first);
    m_fixed_point.values().resize(n);
    auto const policy = m_fixed_point.values().get_execution_policy();
    auto const values = m_fixed_point.values().begin();
    using size_type = std::remove_const_t<decltype(n)>;
    for_each(policy,
        counting_iterator<size_type>(0),
        counting_iterator<size_type>(n),
    [=] P3A_HOST P3A_DEVICE (size_type i) P3A_ALWAYS_INLINE {
      values[i] = unary_op(first[i]);
    });
    return m_fixed_point.compute();
  }
  template <class UnaryOp>
  [[nodiscard]] P3A_NEVER_INLINE
  double transform_reduce(
      subgrid3 grid,
      UnaryOp unary_op)
  {
    m_fixed_point.values().resize(grid.size());
    auto const policy = m_fixed_point.values().get_execution_policy();
    auto const values = m_fixed_point.values().begin();
    for_each(policy, grid,
    [=] P3A_HOST P3A_DEVICE (vector3<int> const& grid_point) P3A_ALWAYS_INLINE {
      int const index = grid.index(grid_point);
      values[index] = unary_op(grid_point);
    });
    return m_fixed_point.compute();
  }
};

template <class T>
using device_associative_sum = 
  associative_sum<T, device_allocator<T>, device_execution>;
template <class T>
using host_associative_sum = 
  associative_sum<T, allocator<T>, serial_execution>;

}
