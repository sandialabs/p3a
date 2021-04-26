#pragma once

#ifdef __CUDACC__
#include <cooperative_groups.h>
#endif

#include "p3a_mpi.hpp"
#include "p3a_int128.hpp"
#include "p3a_quantity.hpp"

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
      T init, BinaryOp binary_op, UnaryOp unary_op) {
    for_each(m_policy, grid,
    [&] (vector3<int> const& item) P3A_ALWAYS_INLINE {
      init = binary_op(std::move(init), unary_op(item));
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

static constexpr int reducer_threads_per_block = 256;
static constexpr int grid_reducer_threads_per_block = 64;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
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
struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

template <class ForwardIt, class T, class BinaryOp, class UnaryOp>
__global__ void cuda_reduce(ForwardIt first, T* g_odata, int n, T init, BinaryOp binop, UnaryOp unop) {
  constexpr int blockSize = reducer_threads_per_block;
  // Handle to thread block group
  cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();
  T* sdata = SharedMemory<T>();
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
  cooperative_groups::sync(cta);
  // do reduction in shared mem
  if (tid < 128) {
    sdata[tid] = myResult = binop(myResult, sdata[tid + 128]);
  }
  cooperative_groups::sync(cta);
  if (tid < 64) {
    sdata[tid] = myResult = binop(myResult, sdata[tid + 64]);
  }
  cooperative_groups::sync(cta);
  cooperative_groups::thread_block_tile<32> tile32 = cooperative_groups::tiled_partition<32>(cta);
  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    myResult = binop(myResult, sdata[tid + 32]);
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      myResult = binop(myResult, tile32.shfl_down(myResult, offset));
    }
  }
  // write result for this block to global mem
  if (cta.thread_rank() == 0) g_odata[blockIdx.x] = myResult;
}

template <class T, class BinaryOp, class UnaryOp>
__global__ void cuda_grid_reduce(
    vector3<int> first, vector3<int> last,
    T* g_odata, T init, BinaryOp binop, UnaryOp unop) {
  constexpr int blockSize = grid_reducer_threads_per_block;
  // Handle to thread block group
  cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();
  T* sdata = SharedMemory<T>();
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
  cooperative_groups::sync(cta);
  // do reduction in shared mem
  cooperative_groups::thread_block_tile<32> tile32 = cooperative_groups::tiled_partition<32>(cta);
  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    myResult = binop(myResult, sdata[tid + 32]);
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      myResult = binop(myResult, tile32.shfl_down(myResult, offset));
    }
  }
  // write result for this block to global mem
  if (cta.thread_rank() == 0) g_odata[block_i] = myResult;
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
  static constexpr int threads_per_block = details::reducer_threads_per_block;
  static constexpr int grid_threads_per_block = details::grid_reducer_threads_per_block;
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
  reducer(reducer&& other) = default;
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

class reproducible_floating_point_adder {
  mpi::comm m_comm;
  device_array<double> m_values;
  reducer<int, device_execution> m_exponent_reducer;
  reducer<int128, device_execution> m_int128_reducer;
 public:
  reproducible_floating_point_adder() = default;
  reproducible_floating_point_adder(
      mpi::comm&& comm_arg)
    :m_comm(std::move(comm_arg))
  {}
  reproducible_floating_point_adder(reproducible_floating_point_adder&&) = default;
  reproducible_floating_point_adder& operator=(reproducible_floating_point_adder&&) = default;
  reproducible_floating_point_adder(reproducible_floating_point_adder const&) = delete;
  reproducible_floating_point_adder& operator=(reproducible_floating_point_adder const&) = delete;
  template <class T, class Dimension, class UnaryOp>
  [[nodiscard]] P3A_NEVER_INLINE
  quantity<T, Dimension> transform_reduce(
      subgrid3 grid,
      quantity<T, Dimension> init,
      UnaryOp unary_op)
  {
    m_values.resize(grid.size());
    auto const policy = m_values.get_execution_policy();
    auto const values = m_values.begin();
    for_each(policy, grid,
    [=] P3A_DEVICE (vector3<int> const& grid_point) P3A_ALWAYS_INLINE {
      int const index = grid.index(grid_point);
      values[index] = unary_op(grid_point).value();
    });
    int const local_max_exponent =
      m_exponent_reducer.transform_reduce(
          m_values.cbegin(), m_values.cend(),
          std::numeric_limits<int>::lowest(),
          maximizes<int>,
    [=] P3A_DEVICE (double const& value) P3A_ALWAYS_INLINE {
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
          m_values.cbegin(), m_values.end(),
          int128::from_double(init.value(), unit),
          adds<int128>,
    [=] P3A_DEVICE (double const& value) P3A_ALWAYS_INLINE {
      return int128::from_double(value, unit);
    });
    int128 global_sum = local_sum;
    auto const int128_mpi_sum_op = 
      mpi::op::create(p3a_mpi_int128_sum);
    m_comm.iallreduce(
        MPI_IN_PLACE,
        &global_sum,
        sizeof(int128),
        MPI_PACKED,
        int128_mpi_sum_op);
    return quantity<T, Dimension>(
        global_sum.to_double(unit));
  }
};

}
