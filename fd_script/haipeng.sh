
rm -rf log
rm -f core*

export NVIDIA_TF32_OVERRIDE=0
export NCCL_ALGO=Tree
export FLAGS_allocator_strategy=auto_growth
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_gemm_use_half_precision_compute_type=False
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
export FLAGS_enable_pir_api=0
export FLAGS_use_append_attn=1

export PYTHONPATH="/root/paddlejob/workspace/env_run/output/zkk/FastDeploy":$PYTHONPATH

# for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
# unset ${name}
# done
# export PADDLE_TRAINER_ID=0
# export PADDLE_TRAINERS_NUM=1
# export TRAINER_INSTANCES_NUM=1
# export TRAINER_INSTANCES=`hostname -i`
# self_ip=`hostname -i`

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

model_path=/root/paddlejob/workspace/env_run/output/45tfp16
model_path=/root/paddlejob/workspace/env_run/output/eb_45_FP8/

# python -m fastdeploy.entrypoints.openai.api_server \
#     --port 8188 \
#     --model ${model_path} \
#     --tensor-parallel-size 8 \
#     --engine-worker-queue-port 8933 \
#     --max-model-len 32768 \
#     --max-num-seqs 256 \
#     --num-gpu-blocks-override 1000 \
#     --max-num-seqs 1

# /opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile \
#     --trace=cuda,nvtx \
#     --force-overwrite true \
#     --capture-range=cudaProfilerApi \
#     -o eb45t_wint8_nvtx_0623_bs97_online \

export LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/output/zkk/lidong/nvshmem/lib:$LD_LIBRARY_PATH
#compute-sanitizer --launch-timeout=0 --tool=memcheck
#/root/paddlejob/workspace/env_run/output/zkk/nsys/bin/nsys profile -c 'cudaProfilerApi' -t "cuda" -f true -o kangkang \
# python -m fastdeploy.entrypoints.openai.api_server \
#     --model ${model_path} \
#     --port 8188 \
#     --metrics-port 8009 \
#     --tensor-parallel-size 1 \
#     --data-parallel-size 8 \
#     --splitwise-role prefill \
#     --engine-worker-queue-port 6678 \
#     --innode-prefill-ports 6677 \
#     --max-model-len 32768 \
#     --max-num-seqs 32 \
#     --gpu-memory-utilization 0.9 \
#     --kv-cache-ratio 0.8 \
#     --enable-prefix-caching \
#     --cache-queue-port 55664 \
#     --quantization wint8 \
#     --num-gpu-blocks-override 5000 \
#     --enable-expert-parallel


# python -m fastdeploy.entrypoints.openai.api_server \
#     --port 8188 \
#     --model ${model_path} \
#     --tensor-parallel-size 1 \
#     --data-parallel-size 8 \
#     --engine-worker-queue-port 6678 \
#     --max-model-len 8192 \
#     --num-gpu-blocks-override 2000 \
#     --max-num-seqs 128 \
#     --quantization wint8 \
#     --metrics-port 8009 \
#     --enable-expert-parallel


export NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0
export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
export NVSHMEM_IBGDA_NUM_RC_PER_PE=1
export NVSHMEM_SYMMETRIC_HEAP_SIZE=32G
export NVSHMEM_IB_ENABLE_IBGDA=true
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_DISABLE_P2P=true
export NVSHMEM_INFO=true
export NVSHMEM_VERSION=true
export NVSHMEM_SYMMETRIC_SIZE=1024
export NVSHMEM_IBGDA_NUM_RC_PER_PE=1  # 注意这里重复了，但按原内容保留
export NVSHMEM_IB_TRAFFIC_CLASS=130
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_ADAPTIVE_ROUTING=1


# 屏蔽平台预设的环境变量，因为框架采用兼容升级，检测到这些配置会使用原方式启动
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
source /root/paddlejob/workspace/env_run/output/zkk/2025_03_28_45T/wuliji.sh
nnodes=$PADDLE_TRAINERS_NUM
rank=$PADDLE_TRAINER_ID

for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
  unset ${name}
done

machines=(0 1 4)
modify_rank=0
for var in "${machines[@]}"; do
  if [[ $rank -eq $var ]]; then
    break
  fi
  ((modify_rank=modify_rank+1))
done
if [[ $modify_rank -eq  ${#machines[@]} ]]; then
    echo "rank ${rank} exit"
    exit 0
fi
rank=$modify_rank
nnodes=${#machines[@]}

source /root/paddlejob/workspace/env_run/output/zkk/erniebot-dev/2024-10-25moe/miniconda3/bin/activate  /root/paddlejob/workspace/env_run/output/zkk/erniebot-dev/2024-10-25moe/miniconda3/envs/deepseekv3_310_new

master=`cat /root/paddlejob/workspace/hostfile | head -n $(($START_RANK+1)) | tail -n 1 | awk '{print $1}'`
port=36677
export python=/root/paddlejob/workspace/env_run/output/zkk/erniebot-dev/2024-10-25moe/miniconda3/envs/deepseekv3_310_new/bin/python

#python -m paddle.distributed.launch --log_dir output/paddle_distributed_logs --gpus ${device} --master $master:$port --nnodes $nnodes --rank $rank \

# python -m fastdeploy.entrypoints.openai.api_server \
#     --port 8188 \
#     --model ${model_path} \
#     --tensor-parallel-size 1 \
#     --data-parallel-size 1 \
#     --engine-worker-queue-port 8933 \
#     --max-model-len 8192 \
#     --num-gpu-blocks-override 1000 \
#     --max-num-seqs 128 \
#     --quantization wint4 \
#     --metrics-port 8009 \
#     --enable-expert-parallel

#/root/paddlejob/workspace/env_run/output/zkk/nsys/bin/nsys profile -c 'cudaProfilerApi' -t "cuda" -f true -o kangkang \

#export FLAGS_use_stride_kernel=0

#compute-sanitizer --launch-timeout=0 --tool=memcheck \
/opt/nvidia/nsight-systems/2024.7.1/bin/nsys profile -c 'cudaProfilerApi' -f true -o kangkang \
python -m fastdeploy.entrypoints.openai.api_server \
       --model ${model_path} \
       --port 9811 --metrics-port 8301 \
       --data-parallel-size 24 --tensor-parallel-size 1 \
       --enable-expert-parallel \
       --max-model-len 32768 \
       --max-num-seqs 24 --kv-cache-ratio 0.85 --num-gpu-blocks-override 1000 \
       --dist-init-ip $master \
       --nnodes $nnodes \
       --node-rank $rank \
        --engine-worker-queue-port 6077 --cache-queue-port 55665 \
        --quantization block_wise_fp8 \
       --scheduler-name "splitwise" \
       --scheduler-host "10.178.5.194" \
       --scheduler-port 6379 \
       --scheduler-ttl 9000 \
       --scheduler-topic zkk_test \
       --scheduler-password "kh^fds9TRj_hePvZ"


# python -m fastdeploy.entrypoints.openai.api_server \
#        --model ${model_path} \
#        --port 9811 --pd-comm-port 2333 --metrics-port 8301 \
#        --data-parallel-size 16 --tensor-parallel-size 1 \
#        --enable-expert-parallel \
#        --max-model-len 32768 \
#        --max-num-seqs 32 --kv-cache-ratio 0.85 --num-gpu-blocks-override 1024 \
#        --ips 10.95.239.100,10.95.239.212 \
#         --engine-worker-queue-port 6377 --cache-queue-port 55665 \
#         --quantization wint4



