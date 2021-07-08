#pragma once

#include "p3a_mpi.hpp"
#include "p3a_int128.hpp"
#include "p3a_quantity.hpp"
#include "p3a_execution.hpp"
#include "p3a_grid3.hpp"
#include "p3a_dynamic_array.hpp"
#include "p3a_counting_iterator.hpp"

namespace p3a {

template <class T>
class minimizer {
 public:
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T operator()(T const& a, T const& b) const {
    return minimum(a, b);
  }
};
template <class T> inline constexpr minimizer<T> minimizes = {};

template <class T>
class maximizer {
 public:
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T operator()(T const& a, T const& b) const {
    return maximum(a, b);
  }
};
template <class T> inline constexpr maximizer<T> maximizes = {};

template <class T>
class adder {
 public:
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T operator()(T const& a, T const& b) const {
    return a + b;
  }
};
template <class T> inline constexpr adder<T> adds = {};

template <class T>
class identity {
 public:
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr T const&
  operator()(T const& a) const {
    return a;
  }
};

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
__device__ P3A_ALWAYS_INLINE T cuda_shuffle_down(T element, unsigned int delta)
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

template <
  class T,
  class Allocator = allocator<double>,
  class ExecutionPolicy = serial_execution>
class reproducible_adder;

/* A reproducible sum of floating-point values.
   this operation is one of the key places where
   a program's output begins to depend on parallel
   partitioning and traversal order, because
   floating-point values do not produce the same
   sum when added in a different order.

   IEEE 754 64-bit floating point format is assumed,
   which has 52 bits in the fraction.

   The idea here is to add the numbers as fixed-point values.
   max_exponent() finds the largest exponent (e) such that
   all values are (<= 2^(e)).
   We then use the value (2^(e - 52)) as the unit, and sum all
   values as integers in that unit.
   This is guaranteed to be at least as accurate as the
   worst-case ordering of the values, i.e. being added
   in order of decreasing magnitude.

   If we used a 64-bit integer type, we would only be
   able to reliably add up to (2^12 = 4096) values
   (64 - 52 = 12).
   Thus we use a 128-bit integer type.
   This allows us to reliably add up to (2^76 > 10^22) values.
   By comparison, supercomputers today
   support a maximum of one million MPI ranks (10^6)
   and each rank typically can't hold more than
   one billion values (10^9), for a total of (10^15) values.
*/

template <
  class Allocator,
  class ExecutionPolicy>
class reproducible_adder<double, Allocator, ExecutionPolicy> {
  mpi::comm m_comm;
  dynamic_array<double, Allocator, ExecutionPolicy> m_values;
  reducer<int, ExecutionPolicy> m_exponent_reducer;
  reducer<int128, ExecutionPolicy> m_int128_reducer;
 public:
  reproducible_adder() = default;
  explicit reproducible_adder(
      mpi::comm&& comm_arg)
    :m_comm(std::move(comm_arg))
  {}
  reproducible_adder(reproducible_adder&&) = default;
  reproducible_adder& operator=(reproducible_adder&&) = default;
  reproducible_adder(reproducible_adder const&) = delete;
  reproducible_adder& operator=(reproducible_adder const&) = delete;
#ifdef __CUDACC__
 public:
#else
 private:
#endif
  [[nodiscard]] P3A_NEVER_INLINE
  double reduce_stored_values()
  {
    int constexpr minimum_exponent =
      std::numeric_limits<int>::lowest();
    int const local_max_exponent =
      m_exponent_reducer.transform_reduce(
          m_values.cbegin(), m_values.cend(),
          minimum_exponent,
          maximizes<int>,
    [=] P3A_HOST P3A_DEVICE (double const& value) P3A_ALWAYS_INLINE {
      if (value == 0.0) return minimum_exponent;
      int exponent;
      std::frexp(value, &exponent);
      return exponent;
    });
    int global_max_exponent = local_max_exponent;
    m_comm.iallreduce(
        &global_max_exponent, 1, mpi::op::max());
    constexpr int mantissa_bits = 52;
    double const unit = std::exp2(
        double(global_max_exponent - mantissa_bits));
    int128 const local_sum =
      m_int128_reducer.transform_reduce(
          m_values.cbegin(), m_values.cend(),
          int128(0),
          adds<int128>,
    [=] P3A_HOST P3A_DEVICE (double const& value) P3A_ALWAYS_INLINE {
      return int128::from_double(value, unit);
    });
    int128 global_sum = local_sum;
    auto const int128_mpi_sum_op = 
      mpi::op::create(p3a_mpi_int128_sum);
    m_comm.iallreduce(
        MPI_IN_PLACE,
        &global_sum,
        sizeof(int128),
        mpi::datatype::predefined_packed(),
        int128_mpi_sum_op);
    return global_sum.to_double(unit);
  }
 public:
  template <class Iterator, class UnaryOp>
  [[nodiscard]] P3A_NEVER_INLINE
  double transform_reduce(
      Iterator first,
      Iterator last,
      UnaryOp unary_op)
  {
    auto const policy = m_values.get_execution_policy();
    auto const values = m_values.begin();
    auto const n = (last - first);
    m_values.resize(n);
    using size_type = std::remove_const_t<decltype(n)>;
    for_each(policy,
        counting_iterator<size_type>(0),
        counting_iterator<size_type>(n),
    [=] P3A_HOST P3A_DEVICE (size_type i) P3A_ALWAYS_INLINE {
      values[i] = unary_op(first[i]);
    });
    return reduce_stored_values();
  }
  template <class UnaryOp>
  [[nodiscard]] P3A_NEVER_INLINE
  double transform_reduce(
      subgrid3 grid,
      UnaryOp unary_op)
  {
    m_values.resize(grid.size());
    auto const policy = m_values.get_execution_policy();
    auto const values = m_values.begin();
    for_each(policy, grid,
    [=] P3A_HOST P3A_DEVICE (vector3<int> const& grid_point) P3A_ALWAYS_INLINE {
      int const index = grid.index(grid_point);
      values[index] = unary_op(grid_point);
    });
    return reduce_stored_values();
  }
};

template <class T>
using device_reproducible_adder = 
  reproducible_adder<T, device_allocator<T>, device_execution>;
template <class T>
using host_reproducible_adder = 
  reproducible_adder<
    T, allocator<T>, serial_execution>;

}
