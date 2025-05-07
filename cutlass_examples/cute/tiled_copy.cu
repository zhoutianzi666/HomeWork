// git reset --hard v3.4.1

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

// This is a simple tutorial showing several ways to partition a tensor into tiles then
// perform efficient, coalesced copies. This example also shows how to vectorize accesses
// which may be a useful optimization or required for certain workloads.
//
// `copy_kernel()` and `copy_kernel_vectorized()` each assume a pair of tensors with
// dimensions (m, n) have been partitioned via `tiled_divide()`.
//
// The result are a part of compatible tensors with dimensions ((M, N), m', n'), where
// (M, N) denotes a statically sized tile, and m' and n' denote the number of such tiles
// within the tensor.
//
// Each statically sized tile is mapped to a CUDA threadblock which performs efficient
// loads and stores to Global Memory.
//
// `copy_kernel()` uses `cute::local_partition()` to partition the tensor and map
// the result to threads using a striped indexing scheme. Threads themselve are arranged
// in a (ThreadShape_M, ThreadShape_N) arrangement which is replicated over the tile.
//
// `copy_kernel_vectorized()` uses `cute::make_tiled_copy()` to perform a similar
// partitioning using `cute::Copy_Atom` to perform vectorization. The actual vector
// size is defined by `ThreadShape`.
//
// This example assumes the overall tensor shape is divisible by the tile size and
// does not perform predication.


/// Simple copy kernel.
//
// Uses local_partition() to partition a tile among threads arranged as (THR_M, THR_N).
template <class TensorS, class TensorD, class ThreadLayout>
__global__ void copy_kernel(TensorS S, TensorD D, ThreadLayout)
{
  using namespace cute;

  // Slice the tiled tensors
  Tensor tile_S = S(make_coord(_,_), blockIdx.x, blockIdx.y);   // (BlockShape_M, BlockShape_N)
  Tensor tile_D = D(make_coord(_,_), blockIdx.x, blockIdx.y);   // (BlockShape_M, BlockShape_N)

  // Construct a partitioning of the tile among threads with the given thread arrangement.

  // Concept:                         Tensor  ThrLayout       ThrIndex
  Tensor thr_tile_S = local_partition(tile_S, ThreadLayout{}, threadIdx.x);  // (ThrValM, ThrValN)
  Tensor thr_tile_D = local_partition(tile_D, ThreadLayout{}, threadIdx.x);  // (ThrValM, ThrValN)

  // Construct a register-backed Tensor with the same shape as each thread's partition
  // Use make_tensor to try to match the layout of thr_tile_S
  Tensor fragment = make_tensor_like(thr_tile_S);               // (ThrValM, ThrValN)

  // Copy from GMEM to RMEM and from RMEM to GMEM
  copy(thr_tile_S, fragment);
  copy(fragment, thr_tile_D);
}

/// Vectorized copy kernel.
///
/// Uses `make_tiled_copy()` to perform a copy using vector instructions. This operation
/// has the precondition that pointers are aligned to the vector size.
///
template <class TensorS, class TensorD, class ThreadLayout, class VecLayout>
__global__ void copy_kernel_vectorized(TensorS S, TensorD D, ThreadLayout, VecLayout)
{

  if (threadIdx.x == 0 && 0) {
    int sm_id;
    // asm volatile("mov.u32 %0, %nsmid;" :"=r"(sm_id));
    asm volatile("mov.u32 %0, %smid;" :"=r"(sm_id));
    printf("sm_id %d\n", sm_id);
  }

  using namespace cute;
  using Element = typename TensorS::value_type;
  
  if (thread0() && 0) {
      printf("%d\n", size<2>(S));
      print(S.stride());
      printf("\n");
      printf("\n");
  }

  // Slice the tensors to obtain a view into each tile.
  Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BlockShape_M, BlockShape_N)
  Tensor tile_D = D(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BlockShape_M, BlockShape_N)

  // Define `AccessType` which controls the size of the actual memory access.
  using AccessType = cutlass::AlignedArray<Element, size(VecLayout{})>;

  // A copy atom corresponds to one hardware memory access.
  using Atom = Copy_Atom<UniversalCopy<AccessType>, Element>;

  // Construct tiled copy, a tiling of copy atoms.
  //
  // Note, this assumes the vector and thread layouts are aligned with contigous data
  // in GMEM. Alternative thread layouts are possible but may result in uncoalesced
  // reads. Alternative vector layouts are also possible, though incompatible layouts
  // will result in compile time errors.
  auto tiled_copy =
    make_tiled_copy(
      Atom{},                       // access size
      ThreadLayout{},               // thread layout
      VecLayout{});                 // vector layout (e.g. 4x1)

  // Construct a Tensor corresponding to each thread's slice.
  auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

  if (thread0()){
    print_latex(tiled_copy);
  }

  Tensor thr_tile_S = thr_copy.partition_S(tile_S);             
  Tensor thr_tile_D = thr_copy.partition_D(tile_D);             

  // Construct a register-backed Tensor with the same shape as each thread's partition
  // Use make_fragment because the first mode is the instruction-local mode
  Tensor fragment = make_fragment_like(thr_tile_D);


  // Copy from GMEM to RMEM and from RMEM to GMEM
  copy(tiled_copy, thr_tile_S, fragment);

  if (thread0()) {
    print(fragment.shape());
    printf("\n");
    print(fragment.stride());
  }

  // if (thread0()) {
  //   printf("%d\n", (int)(size<0>(fragment)));
  //   printf("%d\n", (int)(size<1>(fragment)));
  //   printf("%d\n", (int)(size<2>(fragment)));
  //   float *tmp = (float*)(fragment.data());
  //   for (int i = 0; i < 64; i++) {
  //     printf("%f\n", tmp[i]);
  //   }
  // }

  // copy(tiled_copy, fragment(_,_,_), thr_tile_D(_,_,_));


  //copy(tiled_copy, thr_tile_S, thr_tile_D);
}

/// Main function
int main(int argc, char** argv) {
  // Given a 2D shape, perform an efficient copy
  using namespace cute;
  using Element = float;

  auto tensor_shape = make_shape(128, 64);

  thrust::host_vector<Element> h_S(size(tensor_shape));
  thrust::host_vector<Element> h_D(size(tensor_shape));

  for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i] = static_cast<Element>(i);
    h_D[i] = Element{};
  }

  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_D = h_D;


  // Make tensors
  Tensor tensor_S = make_tensor(make_gmem_ptr(
    thrust::raw_pointer_cast(d_S.data())), 
    make_layout(tensor_shape, LayoutRight{}));
  Tensor tensor_D = 
  make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), 
  make_layout(tensor_shape, LayoutRight{}));

  // Define a statically sized block (M, N).
  // Note, by convention, capital letters are used to represent static modes.
  auto block_shape = make_shape(Int<128>{}, Int<64>{});
  
  // 这他吗也太奇怪了，那如果没有办法整除咋办啊？？

  // if ((size<0>(tensor_shape) % size<0>(block_shape)) || (size<1>(tensor_shape) % size<1>(block_shape))) {
  //   std::cerr << "The tensor shape must be divisible by the block shape." << std::endl;
  //   return -1;
  // }
  // Equivalent check to the above
  // if (not weakly_compatible(block_shape, tensor_shape)) {
  //   std::cerr << "Expected the tensors to be weakly compatible with the block_shape." << std::endl;
  //   return -1;
  // }

  // Tile the tensor (m, n) ==> ((M, N), m', n') where (M, N) is the static tile
  // shape, and modes (m', n') correspond to the number of tiles.
  //
  // These will be used to determine the CUDA kernel grid dimensions.
  Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);      // ((M, N), m', n')
  Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);      // ((M, N), m', n')

  // Thread arrangement
  Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
  
  // Layout thr_layout = Layout< Shape<_32,_8 >,
  //       Stride<_1, _32>> {};

  //print_latex(thr_layout);

  // Vector dimensions
  Layout vec_layout = make_layout(make_shape(Int<1>{}, Int<4>{}), LayoutRight{});

  //print_latex(vec_layout);

  dim3 gridDim (size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));   // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(thr_layout));

  copy_kernel_vectorized<<< gridDim, blockDim >>>(
    tiled_tensor_S,
    tiled_tensor_D,
    thr_layout,
    vec_layout);

  cudaError result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result) << std::endl;
    return -1;
  }

  // Verify
  h_D = d_D;

  int32_t errors = 0;
  int32_t const kErrorLimit = 10;

  for (size_t i = 0; i < h_D.size(); ++i) {
    if (h_S[i] != h_D[i]) {
      std::cerr << "Error. S[" << i << "]: " << h_S[i] << ",   D[" << i << "]: " << h_D[i] << std::endl;

      if (++errors >= kErrorLimit) {
        std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
        return -1;
      }
    }
  }

  std::cout << "Success." << std::endl;

  using CLayout1 =  Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,                              
                                  Stride<Stride<_32,_1>,Stride<_16,_8>>
                                >;
  
  using haha=               Layout<
                    Shape<Shape<_2, _2>, Int<32 / 32>>,
                    Stride<Shape<_1, _2>, _4>>;

  //print_latex(haha{});
  return 0;
}

