project (a.out)
cmake_minimum_required(VERSION 3.1...3.21 FATAL_ERROR)

#file(GLOB all_cu "*.cu")
find_package(CUDA)
include_directories("../../cutlass/include")
include_directories("../../cutlass/")
include_directories("../../cutlass/tools/util/include")

include_directories("../../cutlass/include/")
include_directories("../../../cutlass/include/")



set(CUDA_NVCC_FLAGS -std=c++17 -g -gencode arch=compute_80,code=sm_80;)

#CUDA_ADD_EXECUTABLE(a.out shape_stride_layout.cu)

CUDA_ADD_EXECUTABLE(a.out tiled_copy.cu)
#CUDA_ADD_EXECUTABLE(a.out sgemm_1.cu)
#CUDA_ADD_EXECUTABLE(a.out zeroine_copy.cu)
#CUDA_ADD_EXECUTABLE(a.out demo_shengying.cu)
#CUDA_ADD_EXECUTABLE(a.out issue_1.cu)
#CUDA_ADD_EXECUTABLE(a.out naive_cute_gemm.cu)


target_link_libraries(a.out cublas)

