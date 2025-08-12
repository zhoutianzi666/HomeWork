#include <stdint.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <cuda_fp16.h>

// nvcc elect.cu -arch sm_90


__device__ __forceinline__ uint32_t elect_one_sync(int lane_id) {
    uint32_t pred = 0;
    asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "      elect.sync %%rx|%%px, %2;\n"
      "@%%px mov.s32 %1, 1;\n"
      "      mov.s32 %0, %%rx;\n"
      "}\n"
      : "+r"(lane_id), "+r"(pred)
      : "r"(0xffffffff));
    return pred;
}

__forceinline__ __device__ int get_lane_id() {
    int lane_id;
    asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

__global__ void helloFromGPU (void)
{

    if (elect_one_sync(get_lane_id())) {
        printf("%d\n", threadIdx.x);
    } else {
        printf("not %d\n", threadIdx.x);
    }
}

int main(void)
{
helloFromGPU <<<1, 32>>>();
cudaDeviceReset();
return 0;
}
