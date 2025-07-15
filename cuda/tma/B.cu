#include <stdio.h>
#include <iostream>
#include "cublas_v2.h"

#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda/barrier>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <stdlib.h>

// nvcc B.cu -arch=compute_90 -code=sm_90  -lcuda



#include <cuda.h>         // CUtensormap
#include <cuda/barrier>

using T = half;
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

CUtensorMap tensor_map{};
// rank is the number of dimensions of the array.
constexpr uint32_t rank = 2;
constexpr uint32_t GMEM_WIDTH = 128;
constexpr uint32_t GMEM_HEIGHT = 100;

uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
// The stride is the number of bytes to traverse from the first element of one row to the next.
// It must be a multiple of 16.
uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(T)};

// The box_size is the size of the shared memory buffer that is used as the
// destination of a TMA transfer.
constexpr uint32_t SMEM_WIDTH = 64;
constexpr uint32_t SMEM_HEIGHT = 32;

uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};


__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map) {
  uint64_t start_time = (uint64_t)clock64();
  
  // bluk tensor 的拷贝操作需要 Shared Memory 首地址对齐 128 字节。
  __shared__ alignas(128) T smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

  // 创建 Shared Memory 的 cuda::barrier 变量 
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;

  const bool thread0 = threadIdx.x == 0 && threadIdx.y == 0;

  if (thread0) {
    // 初始化 barrier 
    init(&bar, blockDim.x * blockDim.y);
    // 插入 fence
    cde::fence_proxy_async_shared_cta();    
  }
  __syncthreads();

  int x = 0;
  int y = 0;

  barrier::arrival_token token;
  if (thread0) {
    // 发起 TMA 二维异步拷贝操作
    cde::cp_async_bulk_tensor_2d_global_to_shared(&smem_buffer, &tensor_map, x, y, bar);
    // 设置同步等待点，指定需要等待的拷贝完成的字节数。
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
  } else {
    // Other threads just arrive.
    token = bar.arrive();
  }

  // 等待完成拷贝
  bar.wait(std::move(token));

  if (thread0) {
    for (int i = 0; i < SMEM_HEIGHT; i++) {
      for (int j = 0; j < SMEM_WIDTH; j+= 8) {
        printf("%d\t", (int)(float)smem_buffer[i][j] / 8);
      }
      printf("\n");
    }
  }

  __syncthreads();

  // 插入 fence
  cde::fence_proxy_async_shared_cta();
  __syncthreads();

  if (thread0) {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, x, y, &smem_buffer);
    cde::cp_async_bulk_commit_group();
    cde::cp_async_bulk_wait_group_read<0>();
  }

  if (thread0) {
    (&bar)->~barrier();
  }



uint64_t end_time = (uint64_t)clock64();
if (thread0)
    printf("Overall VM execution time: %lu\n", end_time - start_time);
}

int main(void) {
  
  int buf_len = GMEM_WIDTH * GMEM_HEIGHT;
  thrust::host_vector<T> h_A(buf_len);
  for (int i = 0; i < buf_len; ++i) {
      h_A[i] = i ;
  }
  thrust::device_vector<T> d_A = h_A;
  auto Aptr = thrust::raw_pointer_cast(d_A.data());

  // The distance between elements in units of sizeof(element). A stride of 2
  // can be used to load only the real component of a complex-valued tensor, for instance.
  uint32_t elem_stride[rank] = {1, 1};

  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
    &tensor_map,                // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    rank,                       // cuuint32_t tensorRank,
    Aptr,                 // void *globalAddress,
    size,                       // const cuuint64_t *globalDim,
    stride,                     // const cuuint64_t *globalStrides,
    box_size,                   // const cuuint32_t *boxDim,
    elem_stride,                // const cuuint32_t *elementStrides,
    // Interleave patterns can be used to accelerate loading of values that
    // are less than 4 bytes long.
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    // Swizzling can be used to avoid shared memory bank conflicts.
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    // L2 Promotion can be used to widen the effect of a cache-policy to a wider
    // set of L2 cache lines.
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    // Any element that is outside of bounds will be set to zero by the TMA transfer.
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
  );

  if (res != CUDA_SUCCESS) {
    printf("cuTensorMapEncodeTiled failed: %d\n", res);
    return 1;
  }

  uint3 block = {32,4,1};
  uint3 grid = {1,1,1};

  for(int i = 0; i < 1; i++) {
      kernel <<<grid, block>>>(tensor_map);
  }

  h_A = d_A;

  // for (int i = 0; i < buf_len; ++i) {
  //     printf("%f\n", (float)(h_A[i]));
  // }
  
  return 0;

}