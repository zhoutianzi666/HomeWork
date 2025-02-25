
#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "stdlib.h"
#include <stdio.h>
#include <dlfcn.h>  // dladdr
#include <sys/time.h>
#include <sys/stat.h>
#include "paddle/extension.h"
#include "cuda_profiler_api.h"

std::vector<paddle::Tensor> TimeForward(const paddle::Tensor& x,
                                        const std::string& annotation) {

auto out = paddle::zeros(x.shape(), x.dtype(), paddle::GPUPlace());

  if (annotation == "start") {
    cudaProfilerStart();
  }

  if (annotation == "end") {
    cudaProfilerStop();
  }

  return {out};
}

std::vector<std::vector<int64_t>> TimeInferShape(const std::vector<int64_t>& x_shape) {
    return {x_shape};
}

std::vector<paddle::DataType> TimeInferDtype(const paddle::DataType& x_dtype) {
    return {x_dtype};
}

PD_BUILD_OP(nsys_profile)
    .Inputs({"x"})
    .Attrs({
        "annotation: std::string",
    })
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(TimeForward))
    .SetInferShapeFn(PD_INFER_SHAPE(TimeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TimeInferDtype));

