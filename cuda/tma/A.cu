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

// nvcc A.cu -arch=compute_90 -code=sm_90  -lcuda

using barrier = cuda::barrier<cuda::thread_scope_block>;
static constexpr size_t buf_len = 64*128;

__global__ void add_one_kernel(int* data, size_t offset) {

__syncthreads();
uint64_t start_time = (uint64_t)clock64();

  __shared__ alignas(16) int smem_data[buf_len];

  // 1. a) 用0号线程初始化 barrier，与上面的代码示例类似。
  //    b) 插入一个fence。表示后续执行异步拷贝操作，需要在这个fence之后才执行。
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) { 
    init(&bar, blockDim.x);                                    // a)
    cuda::device::experimental::fence_proxy_async_shared_cta();// b)
  }
  __syncthreads();

  // 2. 发起 TMA 异步拷贝。注意：TMA 操作是用单线程发起。
  if (threadIdx.x == 1) {
    // 3a. 发起异步拷贝
    cuda::memcpy_async(
        smem_data, 
        data + offset, 
        cuda::aligned_size_t<16>(sizeof(smem_data)),
        bar
    );
  }
  // 3b. 所有线程到达该标记点，barrier内部的计数器会加 1。
  barrier::arrival_token token = bar.arrive();
  
  // 3c.等待barrier内部的计数器等于期望数值，即所有线程到达3b点时，当前线程的wait会返回，结束等待。
  bar.wait(std::move(token));

  // 4. 在 Shared Memory 上写数据。
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
    smem_data[i] = 0;
  }

  // 5. 插入fence，保证后续的异步拷贝操作在Shared Memory写数据结束后再启动。
  cuda::device::experimental::fence_proxy_async_shared_cta();   // b)
  __syncthreads();

  // 6. 发起从 Shared Memory 到 Global Memory 的异步拷贝操作。
  if (threadIdx.x == 0) {
    cuda::device::experimental::cp_async_bulk_shared_to_global(
        data + offset, smem_data, sizeof(smem_data));
    // 7. 一种同步方式，创建一个 bulk async-group，异步拷贝在这个 group 中运行，当异步拷贝结束后，
    // group 内部标记为已完成。
    cuda::device::experimental::cp_async_bulk_commit_group();
    // 等待 group 完成。模版参数 0 表示要等待小于等于 0 个 bulk async-group 完成才结束等待。
    cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
  }

__syncthreads();
uint64_t end_time = (uint64_t)clock64();
if (threadIdx.x == 0 || true)
    printf("Overall VM execution time: %lu\n", end_time - start_time);
}

int main(void) {




    using T = int32_t;
    thrust::host_vector<T> h_A(buf_len);
    for (int i = 0; i < buf_len; ++i) {
        h_A[i] = i;
    }
    thrust::device_vector<T> d_A = h_A;
    auto Aptr = thrust::raw_pointer_cast(d_A.data());


    uint3 block = {32,1,1};
    uint3 grid = {1,1,1};


    for(int i = 0; i < 100; i++) {
        add_one_kernel <<<grid, block>>>(Aptr, 0);
    }

    h_A = d_A;

    // for (int i = 0; i < buf_len; ++i) {
    //     printf("%d\n", h_A[i]);
    // }
    
    return 0;
}