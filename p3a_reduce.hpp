#pragma once

namespace p3a {

template <class T>
class minimizer {
 public:
  CPL_ALWAYS_INLINE constexpr
  T operator()(T const& a, T const& b) const {
    return minimum(a, b);
  }
};
template <class T> inline constexpr minimizer<T> minimizes = {};

template <class T>
class adder {
 public:
  CPL_ALWAYS_INLINE constexpr
  T operator()(T const& a, T const& b) const {
    return a + b;
  }
};
template <class T> inline constexpr adder<T> adds = {};

template <class T>
class identity {
 public:
  CPL_ALWAYS_INLINE constexpr T const&
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
  reducer(serial_execution policy, std::ptrdiff_t)
    :m_policy(policy)
  {}
  reducer(reducer&&) = default;
  reducer& operator=(reducer&&) = default;
  reducer(reducer const&) = delete;
  reducer& operator=(reducer const&) = delete;
  template <class BinaryOp, class UnaryOp>
  [[nodiscard]] CPL_NEVER_INLINE
  T transform_reduce(
      subgrid3 grid,
      T init, BinaryOp binary_op, UnaryOp unary_op) {
    for_each(m_policy, grid,
    [&] (vector3<int> const& item) CPL_ALWAYS_INLINE {
      init = binary_op(std::move(init), unary_op(item));
    });
    return init;
  }
};

template <
    class T,
    class BinaryOp,
    class UnaryOp>
[[nodiscard]] CPL_NEVER_INLINE
T transform_reduce(
    serial_execution policy,
    subgrid3 subgrid,
    T init,
    BinaryOp binary_op,
    UnaryOp unary_op)
{
  reducer<T, serial_execution> r(policy, subgrid.volume());
  r.transform_reduce(subgrid, init, binary_op, unary_op);
  return init;
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
  cuda_execution m_policy;
  cuda_device_allocator<T> m_allocator;
  int m_scratch_size;
  T* m_scratch1;
  T* m_scratch2;
  static constexpr int threads_per_block = impl::reducer_threads_per_block;
  static constexpr int grid_threads_per_block = impl::grid_reducer_threads_per_block;
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
    cudaStream_t const cuda_stream = m_policy.cuda_stream();
    impl::cuda_reduce<<<
      dimGrid,
      dimBlock,
      smemSize,
      cuda_stream>>>(first, d_odata, size, init, binop, unop);
  }
  template <class BinaryOp, class UnaryOp>
  void grid_reduction_pass(
      vector3<int> first, vector3<int> last, vector3<int> block_grid,
      T* d_odata, T init, BinaryOp binop, UnaryOp unop) {
    dim3 dimBlock(grid_threads_per_block, 1, 1);
    dim3 dimGrid(block_grid.x(), block_grid.y(), block_grid.z());
    int smemSize = grid_threads_per_block * sizeof(T);
    cudaStream_t cuda_stream = m_policy.cuda_stream();
    impl::cuda_grid_reduce<<<
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
    T host_result = init;
    T const* device_result_ptr = this->reduce_on_device(n, first, init, binop, unop);
    memcpy(m_policy, &host_result, device_result_ptr, sizeof(T));
    return host_result;
  }
};

template <
    class T,
    class BinaryOp,
    class UnaryOp>
[[nodiscard]] CPL_NEVER_INLINE
T transform_reduce(
    cuda_execution policy,
    subgrid3 subgrid,
    T init,
    BinaryOp binary_op,
    UnaryOp unary_op)
{
  reducer<T, cuda_execution> r(policy, subgrid.volume());
  r.transform_reduce(subgrid, init, binary_op, unary_op);
  return init;
}

#endif

}
