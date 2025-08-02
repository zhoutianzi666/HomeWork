#include <cstdio>
#include <cuda_fp16.h>
#include <stdio.h>
#include <iostream>
#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

using T = nv_bfloat16;
// 本cu编译命令 ： nvcc mma.cu -arch=compute_90 -code=sm_90  -o a.out 
// /root/paddlejob/workspace/env_run/output/zkk/nsight-compute/2023.3.1/ncu -o profile -f  --set full ./a.out
__device__ inline void mma(const uint32_t* a_frag,
                           const uint32_t* frag_b,
                           float* frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(frag_b);
  float* c = reinterpret_cast<float*>(frag_c);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

constexpr int MMA_NUMS_EXECUTED_BY_ONE_WARP = 1;
constexpr int32_t WARPS = 8;

__global__ __launch_bounds__(32*WARPS) 
void kernel(const T *x, const T *y, T *z) {

    __shared__ T aTile[16*16];
    __shared__ T bTile[16*8];
    // 输出结果先用stmatrix指令写到shared memory，再从shared memory中写到global memory！
    __shared__ T cTile[16*8];
    

    for (int i = threadIdx.x ; i < 16*16; i += blockDim.x) {
        aTile[i] = x[i];
    }

    for (int i = threadIdx.x ; i < 16*8; i += blockDim.x) {
        bTile[i] = y[i];
    }

    __syncthreads();

    const uint32_t lane_id = threadIdx.x % 32;

    int aTile_index = lane_id % 16 * 16 + lane_id / 16 * 8;
    uint32_t a_register[4];
    uint32_t a_smem = __cvta_generic_to_shared(aTile+aTile_index);
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [ %4 ];\n"
    : "=r"(a_register[0]), "=r"(a_register[1]), "=r"(a_register[2]), "=r"(a_register[3]) 
    : "r"(a_smem)
    );

    //int bTile_index = lane_id % 8 * 16 + lane_id / 8 * 8;
    int bTile_index = lane_id % 16 * 8 + lane_id / 16 * 8;
    uint32_t b_register[2];
    uint32_t b_smem = __cvta_generic_to_shared(bTile+bTile_index);
    asm("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 { %0, %1 }, [ %2 ];\n"
    : "=r"(b_register[0]), "=r"(b_register[1]) 
    : "r"(b_smem)
    );

    float z_register[4] = {0,0,0,0};
    #pragma unroll
    for (int i = 0; i < MMA_NUMS_EXECUTED_BY_ONE_WARP; ++i) {
        mma(a_register, b_register, z_register);
    }
    __syncthreads();


    __nv_bfloat162 val0 = __float22bfloat162_rn({z_register[0], z_register[1]});
    __nv_bfloat162 val1 = __float22bfloat162_rn({z_register[2], z_register[3]});

    int cTile_index = lane_id % 16 * 8;
    uint32_t c_register[2];
    c_register[0] = *(uint32_t*)(&val0);
    c_register[1] = *(uint32_t*)(&val1);
    uint32_t c_smem = __cvta_generic_to_shared(cTile+cTile_index);

    asm volatile ("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n"
        :: "r"(c_smem),
           "r"(c_register[0]), "r"(c_register[1]));


    for (int i = threadIdx.x ; i < 16*8; i += blockDim.x) {
        z[i] = cTile[i];
    }
}



__global__ void run_baseline_gemm(const T *x, const T *y, const T *z_mma) {
    int M = 16;
    int N = 8;
    int K = 16;
    float sum = 0;
    if (threadIdx.x == 0) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                sum = 0;
                for (int k = 0; k < K; k++) {
                    sum += (float)(x[i*K + k]) * (float)(y[k * N + j]);
                }
                float mma_value = (float)(z_mma[i*N + j]);
                if (fabs(sum - mma_value) > 1e-2) {
                    printf("%f\n", mma_value);
                    printf("%f\n", sum);
                }
            }
        }
    }
}



__global__ void init_data_kernel(T *x, int N) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        x[i] = (T)(clock64() % 10 / 10.0);
    }
}


int main() {

    const int32_t grid_num = 1;
    T *x, *y;
    T *z_mma;
    float *z_baseline;

    cudaMalloc(&x, 16*16*sizeof(T));
    cudaMalloc(&y, 16*8*sizeof(T));
    cudaMalloc(&z_mma, 16*8*sizeof(T));
    cudaMalloc(&z_baseline, 16*8*sizeof(float));
    
    init_data_kernel<<<10,128>>>(x, 16*16);
    init_data_kernel<<<10,128>>>(y, 16*8);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm up to cache data into L2
    for (int i = 0; i < 10; ++i) {
        kernel<<<grid_num, WARPS*32>>>(x, y, z_mma);
    }

    const int32_t BENCH_ITER = 10;
    cudaDeviceSynchronize();
     std::cout <<  cudaGetErrorString( cudaGetLastError() ) << std::endl;

    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITER ; ++i) {
        kernel<<<grid_num, WARPS*32>>>(x, y, z_mma);
    }
    cudaEventRecord(stop);

    float time_ms = 0.f;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    double tflops = ( (double)(16*8*16*2) * MMA_NUMS_EXECUTED_BY_ONE_WARP*grid_num*WARPS * BENCH_ITER) 
                    / 1e12
                    / (time_ms / 1e3);
    printf("bf16 tensor core: %fTFlops/s\n", tflops);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    run_baseline_gemm<<<1,1>>>(x, y, z_mma);

    cudaFree(x);
    cudaFree(y);

    return 0;
}


