#include <algorithm>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_with_broadcast.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"


#include "utility.h"

using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    C_DATATYPE,
    int32_t,
    C_DATATYPE,
    C_DATATYPE,
    C_DATATYPE,
    4,
    cutlass::epilogue::thread::ReLu<float>,
    cutlass::multiplies<float>
  >;

  using GemmKernel = 
    typename cutlass::gemm::kernel::DefaultGemmWithBroadcast<
      int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
      int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
      C_DATATYPE, cutlass::layout::RowMajor,
      int32_t,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<8, 8, 16>,
      EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
      2,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;


// A是行存储
// B是列存储
// C是行存储
cudaError_t GemmWithBroadcast(int M, int N, int K, 
                              DATATYPE const *A, int lda, 
                              DATATYPE const *B, int ldb,
                              const BROADCAST_DATATYPE  *broadcast,
                              C_DATATYPE *C, int ldc) {
  cutlass::gemm::GemmCoord problem_size(M, N, K);
  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kBatched,
      problem_size,
      1,
      {(float)1.0, (float)0.f},
      (int8_t *)B, // 输入B
      (int8_t *)A, // 输入A
      (C_DATATYPE *)nullptr,  // 输入C,没啥用哦！因为我把beta搞成0了！
      (C_DATATYPE *)C, // 最终输出
      (C_DATATYPE *)broadcast, // broadcast的tensor
      (C_DATATYPE *)nullptr, // 中间输出
      problem_size.k() * problem_size.n(),
      problem_size.m() * problem_size.k(),
      0,  // 输入C的batch stride
      problem_size.m() * problem_size.n(), //最终输出
      problem_size.n(),  // 广播
      0,                // 中间输出不要他！
      ldb,
      lda,
      0,
      ldc,
      0,                                    // This must be zero，这个是广播的数据吧！
      0 // dont care
    };

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  size_t bytes = Gemm::get_workspace_size(arguments);
  void * workspace;
  cudaMalloc((void**)&workspace, bytes);

  Gemm gemm_op;
  cutlass::Status status = gemm_op.can_implement(arguments);
  status = gemm_op.initialize(arguments, workspace);

  // Launch initialized CUTLASS kernel
  status = gemm_op();
  return cudaSuccess;
}
