# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import triton
import triton.language as tl
from paddle import _C_ops
from paddle.base.framework import OpProtoHolder
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode

from fastdeploy.model_executor.ops.triton_ops.triton_utils import (
    get_dtype_str, paddle_use_triton)

@paddle_use_triton()
def rms_norm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    M,
    N,
    epsilon,
    BLOCK_SIZE_M: tl.constexpr,
    N_npo2: tl.constexpr,
    weight_attr: tl.constexpr,
    bias_attr: tl.constexpr,
):
    row = tl.program_id(axis=0)

    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_an = tl.arange(0, N_npo2)

    # compute var
    all_offs = (row * BLOCK_SIZE_M + offs_am[:, None]) % M * N + offs_an[None, :]

    x_eles = tl.load(x_ptr + all_offs, mask=offs_an[None, :] < N, other=0.0).to(tl.float32)
    var = tl.sum(x_eles * x_eles, axis=1) / N

    resi_hat = x_eles / tl.sqrt(var[:, None] + epsilon)

    if weight_attr:
        weights = tl.load(weight_ptr + offs_an, mask=offs_an < N, other=0.0)
        resi_hat = resi_hat * weights

    if bias_attr:
        bias = tl.load(bias_ptr + offs_an, mask=offs_an < N, other=0.0)
        resi_hat = resi_hat + bias

    tl.store(y_ptr + all_offs, resi_hat, mask=offs_an[None, :] < N)


def rms_norm(x, weight=None, bias=None, epsilon=1e-5):

    assert len(x.shape) == 4, "x should be 4-dim."
    weight_attr = 0
    weight = None
    if weight is not None:
        assert len(weight.shape) == 1, "weight should be 1-dim"
        assert weight.shape[-1] == x.shape[-1], "x and weight should have same shape[-1]"
        weight_attr = 1
    bias_attr = 1
    bias_attr = 1

    M = x.shape[0] * x.shape[1] * x.shape[2]
    N = x.shape[3]
    N_npo2 = triton.next_power_of_2(N)

    BLOCK_SIZE_M = 1
    y = paddle.empty_like(x)

    grid = ((M+BLOCK_SIZE_M-1)//BLOCK_SIZE_M,)
    grid = ("(M+BLOCK_SIZE_M-1)/BLOCK_SIZE_M",)
    rms_norm_kernel[grid](
        x,
        y,
        weight,
        bias,
        M,
        N,
        epsilon,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        N_npo2=N_npo2,
        weight_attr=weight_attr,
        bias_attr=bias_attr,
    )

    return y



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import paddle

batch = 2
seq = 36000
num_heads = 1
head_dim = 64*30
dtype= "float16"
x = paddle.rand([batch, seq, num_heads, head_dim], dtype=dtype)
weight = paddle.rand([head_dim], dtype=dtype)
bias = paddle.rand([head_dim], dtype=dtype)

for i in range(100):
    baseline = paddle.incubate.nn.functional.fused_rms_norm(x, weight, bias, 1e-5, begin_norm_axis=3)

for i in range(100):
    mt_result = rms_norm(x,weight,bias,1e-5)


baseline = baseline[0]
print(paddle.max(paddle.abs(baseline-mt_result)))
