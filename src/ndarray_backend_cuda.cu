#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides
__device__ size_t index(size_t gid, CudaVec shape, CudaVec strides,
                        size_t offset) {
  for (ssize_t i = shape.size - 1; i >= 0; --i) {
    offset += gid % shape.data[i] * strides.data[i];
    gid /= shape.data[i];
  }
  return offset;
}



__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid < size)
    out[gid] = a[index(gid, shape, strides, offset)];
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}



__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out,
                                   size_t size, CudaVec shape, CudaVec strides,
                                   size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[index(gid, shape, strides, offset)] = a[gid];
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size,
                                              VecToCuda(shape),
                                              VecToCuda(strides), offset);
  /// END SOLUTION
}



__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size,
                                    CudaVec shape, CudaVec strides,
                                    size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[index(gid, shape, strides, offset)] = val;
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, out->size,
                                               VecToCuda(shape),
                                               VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

#define DEFINE(sym, b_param, b_kparam, b_karg, op) \
__global__ void sym##Kernel(const scalar_t* a, b_kparam scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = op; \
} \
 \
void sym(const CudaArray& a, b_param CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  sym##Kernel<<<dim.grid, dim.block>>>(a.ptr, b_karg out->ptr, out->size); \
}
#define COMMA ,

#define DEFINE_UNARY(sym, op) DEFINE(Ewise##sym, ,,, op(a[gid]))
#define DEFINE_EWISE_BINARY(sym, op) \
  DEFINE(Ewise##sym, const CudaArray& b COMMA, const scalar_t *b COMMA, \
         b.ptr COMMA, op)

#define DEFINE_SCALAR(sym, op) \
  DEFINE(Scalar##sym, scalar_t val COMMA, scalar_t val COMMA, val COMMA, op)
#define DEFINE_PREFIX_SCALAR(sym, op) DEFINE_SCALAR(sym, op(a[gid], val))

#define DEFINE_INFIX(sym, op) \
  DEFINE_EWISE_BINARY(sym, a[gid] op b[gid]) \
  DEFINE_SCALAR(sym, a[gid] op val)
#define DEFINE_PREFIX(sym, op) \
  DEFINE_EWISE_BINARY(sym, op(a[gid], b[gid])) \
  DEFINE_PREFIX_SCALAR(sym, op)

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

DEFINE_INFIX(Add, +)
DEFINE_INFIX(Mul, *)
DEFINE_INFIX(Div, /)
/* Prefix operators will break when we change the type of scalar_t. */
DEFINE_PREFIX_SCALAR(Power, powf)
DEFINE_PREFIX(Maximum, fmax)
DEFINE_INFIX(Eq, ==)
DEFINE_INFIX(Ge, >=)
DEFINE_UNARY(Log, logf)
DEFINE_UNARY(Exp, expf)
DEFINE_UNARY(Tanh, tanhf)

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

#define TILE_THREAD 8
#define TILES_PER_THREAD 1
#define TILE_BLOCK 64
__global__ void MatmulKernel(const scalar_t *a, const scalar_t *b,
                             scalar_t *out, int m, int n, int p) {
  __shared__ scalar_t a_shmem[TILE_BLOCK][TILE_BLOCK],
                      b_shmem[TILE_BLOCK][TILE_BLOCK],
                      out_shmem[TILE_BLOCK][TILE_BLOCK];
  scalar_t a_regs[TILE_THREAD][TILE_THREAD],
           b_regs[TILE_THREAD][TILE_THREAD],
           out_regs[TILE_THREAD][TILE_THREAD];

  for (int tile_x = 0; tile_x < TILES_PER_THREAD; ++tile_x)
    for (int tile_y = 0; tile_y < TILES_PER_THREAD; ++tile_y) {
      int ii = (threadIdx.x + tile_x) * TILE_THREAD;
      int kk = (threadIdx.y + tile_y) * TILE_THREAD;
      for (int i = 0; i < TILE_THREAD; ++i)
        for (int k = 0; k < TILE_THREAD; ++k)
          out_shmem[ii + i][kk + k] = 0.;
    }
  int iii = blockIdx.x * TILE_BLOCK;
  int kkk = blockIdx.y * TILE_BLOCK;
  for (int jjj = 0; jjj < n; jjj += TILE_BLOCK) {
    // Read block tiles into shared memory, zero-padding as necessary.
    for (int tile_x = 0; tile_x < TILES_PER_THREAD; ++tile_x)
      for (int tile_y = 0; tile_y < TILES_PER_THREAD; ++tile_y) {
        // Because we're working with square block tiles, thread indices may be
        // used interchangeably for the jth dimension.
        int ii = (threadIdx.x + tile_x) * TILE_THREAD;
        int kk = (threadIdx.y + tile_y) * TILE_THREAD;
        for (int i = 0; i < TILE_THREAD; ++i)
          for (int k = 0; k < TILE_THREAD; ++k) {
            a_shmem[ii + i][kk + k] = iii + ii + i < m && jjj + kk + k < n
                                      ? a[(iii + ii + i) * n + jjj + kk + k]
                                      : 0.;
            b_shmem[ii + i][kk + k] = jjj + ii + i < n && kkk + kk + k < p
                                      ? b[(jjj + ii + i) * p + kkk + kk + k]
                                      : 0.;
          }
      }
    __syncthreads();
    // Multiply thread tiles.
    for (int tile_x = 0; tile_x < TILES_PER_THREAD; ++tile_x)
      for (int tile_y = 0; tile_y < TILES_PER_THREAD; ++tile_y) {
        // Zero output thread tile.
        for (int i = 0; i < TILE_THREAD; ++i)
          for (int j = 0; j < TILE_THREAD; ++j)
            out_regs[i][j] = 0.;

        int ii = (threadIdx.x + tile_x) * TILE_THREAD;
        int kk = (threadIdx.y + tile_y) * TILE_THREAD;
        for (int jj = 0; jj < TILE_BLOCK; jj += TILE_THREAD) {
          // Read thread tiles into registers.
          for (int j = 0; j < TILE_THREAD; ++j) {
            for (int i = 0; i < TILE_THREAD; ++i)
              a_regs[i][j] = a_shmem[ii + i][jj + j];
            for (int k = 0; k < TILE_THREAD; ++k)
              b_regs[j][k] = b_shmem[jj + j][kk + k];
          }
          // Multiply.
          for (int i = 0; i < TILE_THREAD; ++i)
            for (int k = 0; k < TILE_THREAD; ++k)
              for (int j = 0; j < TILE_THREAD; ++j)
                out_regs[i][k] += a_regs[i][j] * b_regs[j][k];
        }
        // Aggregate thread tile into shared memory.
        for (int i = 0; i < TILE_THREAD; ++i)
          for (int k = 0; k < TILE_THREAD; ++k)
            out_shmem[ii + i][kk + k] += out_regs[i][k];
      }
  }
  __syncthreads();
  // Write block tiles to global memory.
  for (int tile_x = 0; tile_x < TILES_PER_THREAD; ++tile_x)
    for (int tile_y = 0; tile_y < TILES_PER_THREAD; ++tile_y) {
      int ii = (threadIdx.x + tile_x) * TILE_THREAD;
      int kk = (threadIdx.y + tile_y) * TILE_THREAD;
      for (int i = 0; i < TILE_THREAD && iii + ii + i < m; ++i)
        for (int k = 0; k < TILE_THREAD && kkk + kk + k < p; ++k)
          out[(iii + ii + i) * p + kkk + kk + k] = out_shmem[ii + i][kk + k];
    }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  dim3 gridDims((M + TILE_BLOCK - 1) / TILE_BLOCK,
                (P + TILE_BLOCK - 1) / TILE_BLOCK),
       blockDims(TILE_BLOCK / TILE_THREAD / TILES_PER_THREAD,
                 TILE_BLOCK / TILE_THREAD / TILES_PER_THREAD);

  MatmulKernel<<<gridDims, blockDims>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
}
#undef TILE_BLOCK
#undef TILES_PER_THREAD
#undef TILE_THREAD

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

// A CUDA block reduces.
#define SCALE 2
#define BLOCKDIM 1024
__global__ void ReduceMaxKernel(const scalar_t *a, scalar_t *out,
                                size_t reduce_size) {
  __shared__ scalar_t inters[BLOCKDIM / 2];

  assert(BLOCKDIM == blockDim.x);
  // Reduce in the thread.
  size_t reductions = (reduce_size + blockDim.x - 1) / blockDim.x,
         idx_thread = threadIdx.x * reductions,
         idx_block = blockIdx.x * reduce_size;
  scalar_t acc = -INFINITY;
  for (size_t i = idx_thread; i < idx_thread + reductions && i < reduce_size;
       ++i)
    if (a[idx_block + i] > acc)
      acc = a[idx_block + i];
  for (size_t factor = 1; factor < BLOCKDIM; factor *= SCALE) {
    if (threadIdx.x % factor == 0 && threadIdx.x % (factor * SCALE) != 0)
      inters[threadIdx.x / 2] = acc;
    __syncthreads();
    if (threadIdx.x % (factor * SCALE) == 0)
      if (inters[(threadIdx.x + factor) / 2] > acc)
        acc = inters[(threadIdx.x + factor) / 2];
  }
  if (!threadIdx.x)
    out[blockIdx.x] = acc;
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  dim3 gridDims(out->size), blockDims(BLOCKDIM);
  ReduceMaxKernel<<<gridDims, blockDims>>>(a.ptr, out->ptr, reduce_size);
  /// END SOLUTION
}


__global__ void ReduceSumKernel(const scalar_t *a, scalar_t *out,
                                size_t reduce_size) {
  __shared__ scalar_t inters[BLOCKDIM / 2];

  assert(BLOCKDIM == blockDim.x);
  // Reduce in the thread.
  size_t reductions = (reduce_size + blockDim.x - 1) / blockDim.x,
         idx_thread = threadIdx.x * reductions,
         idx_block = blockIdx.x * reduce_size;
  scalar_t acc = 0.;
  for (size_t i = idx_thread; i < idx_thread + reductions && i < reduce_size;
       ++i)
    acc += a[idx_block + i];
  for (size_t factor = 1; factor < BLOCKDIM; factor *= SCALE) {
    if (threadIdx.x % factor == 0 && threadIdx.x % (factor * SCALE) != 0)
      inters[threadIdx.x / 2] = acc;
    __syncthreads();
    if (threadIdx.x % (factor * SCALE) == 0)
      acc += inters[(threadIdx.x + factor) / 2];
  }
  if (!threadIdx.x)
    out[blockIdx.x] = acc;
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  dim3 gridDims(out->size), blockDims(BLOCKDIM);
  ReduceSumKernel<<<gridDims, blockDims>>>(a.ptr, out->ptr, reduce_size);
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
