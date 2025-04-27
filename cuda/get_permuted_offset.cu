

#include <cstdio>
#include <cuda_fp16.h>
#include <stdio.h>
#include <iostream>
#include "/root/paddlejob/workspace/env_run/output/zkk/2025_03_28_45T/baidu/paddle_internal/EfficientLLM/custom_ops/gpu_ops/append_attn/mem_util.cuh"

// nvcc get_permuted_offset.cu -arch=compute_90 -code=sm_90 && ./a.out 



#define dtype half

__global__ void init_data_kernel(dtype *x, int N) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        x[i] = __float2half(i * 1.0);
    }
}

__global__ void kernel2(dtype *x, int N) {
  
  const int BYTES = 16;
  const int threads = 32;
  const int num_per_thread = BYTES / sizeof(dtype);
  __shared__ dtype aTile[threads * num_per_thread];
  int aTile_index = threadIdx.x * num_per_thread;

  int tid = threadIdx.x;

  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(aTile + aTile_index));
  dtype* glob_ptr = x + aTile_index;

  smem_t zkk_smem(aTile);
  uint32_t kv_smem_offset_w1 = smem_t::get_permuted_offset<128/8>(tid / 8, tid % 8);
  uint32_t kv_smem_offset_w2 = smem_t::get_permuted_offset<128/8>(tid / 8, tid % 8 + 8);
  
  printf("%d, %d\n", tid, kv_smem_offset_w2 - kv_smem_offset_w1);

  asm volatile(
    "{\n"
    " cp.async.cg.shared.global [%0], [%1], %2, 16;\n"
    "cp.async.commit_group;\n"
    "cp.async.wait_group 0\n;"
    "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
  
  if (threadIdx.x == 1) {
    for (int i = 0; i < threads * num_per_thread; i++) {
        printf("%f \n", (float)(aTile[i]));
    }
  }
}

int main() {
    const int N_DATA = 1024 * 1024 * 128;
    dtype *x;
    cudaMalloc(&x, N_DATA * sizeof(dtype));
    int blocksize = 128;
    int grid = 1;
    init_data_kernel<<<grid, blocksize>>>(x, N_DATA);
    
    blocksize = 32;
    kernel2<<<grid, blocksize>>>(x, N_DATA);  
    cudaDeviceSynchronize();
    std::cout <<  cudaGetErrorString( cudaGetLastError() ) << std::endl;
    cudaFree(x);
    return 0;
}


