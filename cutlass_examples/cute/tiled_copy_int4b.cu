#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <bitset>
#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

template <class TensorS, class TensorD, class ThreadLayout, class VecLayout>
__global__ void copy_kernel_vectorized(TensorS S, TensorD D, ThreadLayout, VecLayout) {
  using namespace cute;
  using Element = typename TensorS::value_type;


  Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y);
  Tensor tile_D = D(make_coord(_, _), blockIdx.x, blockIdx.y);

  // Define `AccessType` which controls the size of the actual memory access.
  using AccessType = cutlass::AlignedArray<Element, size(VecLayout{})>;

  // A copy atom corresponds to one hardware memory access.
  // using Atom = Copy_Atom<UniversalCopy<AccessType>, Element>;
  using Atom = Copy_Atom<DefaultCopy, Element>;

  auto tiled_copy = make_tiled_copy(
      Atom{},
      ThreadLayout{},
      VecLayout{});

  auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

  Tensor thr_tile_S = thr_copy.partition_S(tile_S);             
  Tensor thr_tile_D = thr_copy.partition_D(tile_D);             

  Tensor fragment = make_fragment_like(thr_tile_D);

  copy(tiled_copy, thr_tile_S, fragment);

  if (threadIdx.x == 2) {
    print(fragment.shape());
    printf("\n");
    print(fragment.stride());
    printf("\n");
    print(fragment((_,_),0,0));
    printf("\n");
    print_tensor(fragment((_,_),0,0));
    printf("\n");

    auto tmp = (uint8_t*)(raw_pointer_cast(fragment.data()));



    for (int j = 0; j < 2; j++) {
      printf("%d\n", tmp[j]);
    }
  }

  copy(tiled_copy, fragment(_,_,_), thr_tile_D(_,_,_));

  // 直接拷贝一下！
  // copy(tiled_copy, thr_tile_S, thr_tile_D);
}

/// Main function
int main(int argc, char** argv) {
  using namespace cute;
  //using Element = int8_t;
  using Element = cutlass::int4b_t;

  // cutlass::int4b_t aa;
  // printf("%d\n", static_cast<int>(aa));
  // exit(0);

  auto tensor_shape = make_shape(128, 64);

  thrust::host_vector<Element> h_S(size(tensor_shape));
  thrust::host_vector<Element> h_D(size(tensor_shape));

  for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i].storage = (i % 256 + 32) & (0x000000ff);
    h_D[i] = Element{};
  }
  
  // 下面这句话证明了！
  auto num = *((int*)(h_S.data()));
  std::cout << std::bitset<32>(num) << std::endl;

  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_D = h_D;

  Tensor tensor_S = make_tensor(make_gmem_ptr(
    thrust::raw_pointer_cast(d_S.data())), 
    make_layout(tensor_shape, LayoutRight{}));
  Tensor tensor_D = 
  make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), 
  make_layout(tensor_shape, LayoutRight{}));


  auto block_shape = make_shape(Int<128>{}, Int<64>{});

  // Tile the tensor (m, n) ==> ((M, N), m', n') where (M, N) is the static tile
  // shape, and modes (m', n') correspond to the number of tiles.
  //
  // These will be used to determine the CUDA kernel grid dimensions.
  Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);      // ((M, N), m', n')
  Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);      // ((M, N), m', n')

  Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
  Layout vec_layout = make_layout(make_shape(Int<1>{}, Int<2>{}), LayoutRight{});

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

  h_D = d_D;

  int32_t errors = 0;
  int32_t const kErrorLimit = 10;

  num = *((int*)(h_D.data()));
  std::cout << std::bitset<32>(num) << std::endl;
  //exit(0);

  for (size_t i = 0; i < h_D.size(); ++i) {
    if (h_S[i] != h_D[i]) {
      std::cerr << "Error. S[" << i << "]: " << h_S[i] << ",   D[" << i << "]: " << h_D[i] << std::endl;

      if (++errors >= kErrorLimit) {
        std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
        return -1;
      }
    }
  }
  return 0;
}
