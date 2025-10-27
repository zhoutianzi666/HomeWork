# Copyright (c) 2023  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import gc
import os
import json
import tqdm
import paddle
from .hadamard_utils import random_hadamard_matrix



def get_orthogonal_matrix(size, mode, device="cuda"):
    """
    获取一个正交矩阵，可以是随机生成的、哈达马尔矩阵或者哈达马尔矩阵的FFN2版本。
    
    Args:
        size (int): 正交矩阵的大小。
        mode (str, optional): 生成方式，可选值为"random", "hadamard", "hadamard_ffn2"中的一种。默认为"random"。
            - "random"：生成一个随机正交矩阵。
            - "hadamard"：生成一个哈达马尔矩阵。
            - "hadamard_ffn2"：生成一个哈达马尔矩阵的FFN2版本。
        device (str, optional): 设备类型，默认为"cuda"。
    
    Returns:
        paddle.Tensor: 返回一个大小为size*size的正交矩阵，维度为2。
    
    Raises:
        ValueError: 如果mode不在"random", "hadamard", "hadamard_ffn2"中。
    """
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    elif mode == "hadamard_ffn2":
        return random_hadamard_matrix(size, device, True)
    else:
        raise ValueError(f"Unknown mode {mode}")


Q_ffn2, moe_block_size = get_orthogonal_matrix(896, "hadamard_ffn2") # weight shape: [896, 8192], Q_ffn2 shape: [896, 896], moe_block_size 128
Q_ffn2=Q_ffn2.cast("float32")

# 推理的快速hadamard变化比这个Q_ffn2 要大 (block_size**0.5)倍 即 Q_ffn2=Q_ffn2*(block_size**0.5)，推理对精度用这个!!!
# Q_ffn2 就是hadamard_ffn2矩阵，输入构造 X  X @ Q_ffn2 然后  Q_ffn2.cast("float32").T @ weight
