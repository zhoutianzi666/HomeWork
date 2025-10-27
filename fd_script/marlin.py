import paddle
from paddle.distributed import fleet

strategy = fleet.DistributedStrategy()
strategy.hybrid_configs = {
    "dp_degree": 1,
    "mp_degree": 2,
    "pp_degree": 1,
    "sharding_degree": 1,
}



fleet.init(is_collective=True, strategy=strategy)


for i in range(100):
    out = paddle.randn([100,100])

    out = out +1

    import paddle.distributed as dist

    dist.all_reduce(out)
    continue

    
    paddle.distributed.stream.all_reduce(out, op=dist.ReduceOp.SUM, 
                                             sync_op=True, 
                                             use_calc_stream=True)

print(out)

exit(0)



import torch
paddle.seed(100)
import fastdeploy
from fastdeploy.model_executor.ops.gpu import MoeWna16MarlinGemmApi
from fastdeploy.model_executor.ops.gpu import tritonmoe_preprocess

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_moe_permute_scales
num_experts = 64
hidden_size = 8192
moe_intermediate_size = 3584 // 8

M = 4
hidden_states = paddle.randn([M, hidden_size], dtype='bfloat16')

def get_weight(K,N):
    old_weight = paddle.randn([num_experts, K, N], dtype='bfloat16') * 4
    old_weight = old_weight.cast("int32")
    old_weight[old_weight > 7] = 7
    old_weight[old_weight < -7] = -7
    
    uint_old_weight = old_weight + 8
    uint_old_weight = uint_old_weight.cast("bfloat16")

    uint_old_weight = uint_old_weight.cast("int32")
    uint_old_weight = uint_old_weight.reshape([0,-1,8,N])
    res = paddle.zeros([num_experts, K // 8, N], dtype='int32')
    for j in range(8):
        tmp = uint_old_weight.cast("int32")[:,:,j,:]
        res = res | (tmp << (j*4))

    tmp = res.numpy()
    w13_qweight = torch.from_numpy(tmp).cuda()
    w13_g_idx_sort_indices = torch.empty((num_experts, 0), dtype=torch.int32, device=w13_qweight.device)
    marlin_w13_qweight = ops.gptq_marlin_moe_repack(
        w13_qweight,
        w13_g_idx_sort_indices,
        w13_qweight.shape[1] * 8,
        w13_qweight.shape[2],
        4,
    )
    w1 = paddle.to_tensor(marlin_w13_qweight.cpu().numpy())
    #w1 = w1[0].tile([64,1,1])

    tmp = 0x77777777
    b_zeros = paddle.zeros([num_experts, 1, N // 8], dtype='bfloat16').cast("int32") + tmp


    old_w1_scale = paddle.randn([num_experts, 1, N], dtype='bfloat16') * 0.01
    tmp = old_w1_scale.numpy()
    w1_scale = torch.from_numpy(tmp).cuda().view(torch.bfloat16)
    w1_scale = marlin_moe_permute_scales(w1_scale,
                                         size_k=-1, #useless
                                         size_n=N,
                                         group_size=-1)

    w1_scale = w1_scale.view(torch.uint16)
    w1_scale = paddle.to_tensor(w1_scale.cpu().numpy())
    #w1_scale = w1_scale[0].tile([64,1,1])

    return w1, old_weight, b_zeros, w1_scale, old_w1_scale

w1, old_w1, b1_zeros, w1_scale, old_w1_scale = get_weight(hidden_size, moe_intermediate_size * 2)

w2, old_w2, b2_zeros, w2_scale, old_w2_scale = get_weight(moe_intermediate_size, hidden_size)

b1_zeros = None
b2_zeros = None

gating_output = paddle.randn([M, num_experts], dtype='bfloat16')
scores = paddle.nn.functional.softmax(gating_output, axis=-1).astype("float32")

block_size_m = 64
topk = 8
topk_weights, topk_ids = paddle.topk(scores, k=topk,axis=-1,sorted=False)

workspace = paddle.empty([528], dtype="int32")



sorted_token_ids, expert_ids, num_tokens_post_padded = tritonmoe_preprocess(topk_ids, num_experts, block_size_m)

gemm_out = MoeWna16MarlinGemmApi(hidden_states,
                                 c_or_none=None,
                                 b_q_weight=w1,
                                 b_scales=w1_scale,
                                 global_scale_or_none=None,
                                 b_zeros_or_none=b1_zeros,
                                 g_idx_or_none=None,
                                 perm_or_none=None,
                                 workspace=workspace,
                                 sorted_token_ids=sorted_token_ids,
                                 expert_ids=expert_ids,
                                 num_tokens_post_padded=num_tokens_post_padded,
                                 topk_weights=topk_weights,
                                 moe_block_size=block_size_m,
                                 top_k=topk,
                                 mul_topk_weights=False,
                                 is_ep=False,
                                 b_q_type_str="uint4",
                                 size_m=M,
                                 size_n= moe_intermediate_size * 2,
                                 size_k=hidden_size,
                                 is_k_full=True,
                                 use_atomic_add=True,
                                 use_fp32_reduce=True,
                                 is_zp_float=False)[0]

swiglu_out = paddle.incubate.nn.functional.swiglu(gemm_out)

gemm_out = MoeWna16MarlinGemmApi(swiglu_out,
                                 c_or_none=None,
                                 b_q_weight=w2,
                                 b_scales=w2_scale,
                                 global_scale_or_none=None,
                                 b_zeros_or_none=b2_zeros,
                                 g_idx_or_none=None,
                                 perm_or_none=None,
                                 workspace=workspace,
                                 sorted_token_ids=sorted_token_ids,
                                 expert_ids=expert_ids,
                                 num_tokens_post_padded=num_tokens_post_padded,
                                 topk_weights=topk_weights,
                                 moe_block_size=block_size_m,
                                 top_k=1,
                                 mul_topk_weights=True,
                                 is_ep=False,
                                 b_q_type_str="uint4",
                                 size_m=M * topk,
                                 size_n= hidden_size,
                                 size_k=moe_intermediate_size,
                                 is_k_full=True,
                                 use_atomic_add=True,
                                 use_fp32_reduce=True,
                                 is_zp_float=False)[0]

gemm_out.reshape_([M,-1,hidden_size])
out = gemm_out.sum(axis=1)
print(out)



# baseline = hidden_states @ (old_w1[0,:,:].cast("bfloat16") * old_w1_scale[0,:,:])

# print(baseline[0][:20])
# print(gemm_out[0][:20])




new1 = old_w1.cast("bfloat16") * old_w1_scale
new2 = old_w2.cast("bfloat16") * old_w2_scale


from fastdeploy.model_executor.ops.gpu import moe_expert_dispatch
(
    permute_input,
    token_nums_per_expert,
    permute_indices_per_token,
    new_topk_weights,
    new_topk_idx,
    expert_idx_per_token,
) = moe_expert_dispatch(
    hidden_states,
    scores,
    None,
    None,
    topk,
    False,
    topk_only_mode=True,
)

expert_idx_per_token = None
ffn_out = fastdeploy.model_executor.ops.gpu.moe_expert_ffn(
            permute_input,
            token_nums_per_expert,
            new1,
            new2,
            None,
            None,
            None,
            None,
            None,
            "",
            False)

from fastdeploy.model_executor.ops.gpu import moe_expert_reduce

# reduce 中会做 topk 个 weight 的 norm 和 routed_scaling_factor
fused_moe_out = moe_expert_reduce(
    ffn_out,
    new_topk_weights,
    permute_indices_per_token,
    new_topk_idx,
    None,
    norm_topk_prob=False,
    routed_scaling_factor=1.0)


print(fused_moe_out)
print(out)




