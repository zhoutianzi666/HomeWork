
# 屏蔽平台预设的环境变量，因为框架采用兼容升级，检测到这些配置会使用原方式启动
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
source /root/paddlejob/workspace/env_run/output/zkk/2025_03_28_45T/wuliji.sh
export PATH=/usr/mpi/gcc/openmpi-4.1.5a1/bin:$PATH
export NCCL_DEBUG=
export PATH=/usr/local/cuda/bin/:$PATH

nnodes=$PADDLE_TRAINERS_NUM
rank=$PADDLE_TRAINER_ID

for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
  unset ${name}
done

START_RANK=0
END_RANK=$nnodes
END_RANK=1


if [[ $rank -lt $START_RANK ]]; then
    echo "rank exit"
    exit 0
fi

if [[ $rank -ge $END_RANK ]]; then
    echo "rank exit"
    exit 0
fi


rank=$(($rank-$START_RANK))
nnodes=$(($END_RANK-$START_RANK))
master=`cat /root/paddlejob/workspace/hostfile | head -n $(($START_RANK+1)) | tail -n 1 | awk '{print $1}'`
port=36678
export FLAGS_hardamard_moe_block_size=128

set -ex
source /root/paddlejob/workspace/env_run/output/zkk/erniebot-dev/2024-10-25moe/miniconda3/bin/activate  /root/paddlejob/workspace/env_run/output/zkk/erniebot-dev/2024-10-25moe/miniconda3/envs/deepseekv3_310_new

#export PYTHONPATH="/root/paddlejob/workspace/env_run/output/zkk/triton_to_static/PaddleMIX":$PYTHONPATH

#export PYTHONPATH="/root/paddlejob/workspace/env_run/output/zkk/2025_05_28_fefactor/baidu/paddle_internal/FastDeploy":$PYTHONPATH

export PYTHONPATH="/root/paddlejob/workspace/env_run/output/zkk/2025_05_28_fefactor/baidu/paddle_internal/fc_06_21_23_05/baidu/paddle_internal/FastDeploy":$PYTHONPATH



export PYTHONPATH="/root/paddlejob/workspace/env_run/output/zkk/2025_03_28_45T/PaddleNLP":$PYTHONPATH


rm -rf log*
rm -f core*

export NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=xgbe0
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

export FLAGS_call_stack_level=2
export GLOG_logtostderr=true
export GLOG_v=0
export NVIDIA_TF32_OVERRIDE=0
export NCCL_ALGO=Tree
export FLAGS_allocator_strategy=auto_growth
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_gemm_use_half_precision_compute_type=False
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
export FLAGS_enable_pir_api=0
export FLAGS_use_append_attn=1

#export LD_LIBRARY_PATH=/usr/local/cuda-12.3/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH

export device=0,1,2,3,4,5,6,7
#export device=0,1,2,3
export USE_WORKER_V1=1
rm -rf log*
model_path="/root/paddlejob/workspace/env_run/output/fastdeploy_45t"
#model_path="/root/paddlejob/workspace/env_run/output/chenjianye/eb45t02"
#model_path="/root/paddlejob/workspace/env_run/output/kaiyuan_45Tw4a8quant"

# for bs in {1..1..1}; do
#   #/root/paddlejob/workspace/env_run/output/zkk/nsys/bin/nsys profile -c 'cudaProfilerApi' -t "cuda" -f true -o kangkang_$bs \
#   python -m paddle.distributed.launch --log_dir output/paddle_distributed_logs --gpus ${device} --master $master:$port --nnodes $nnodes --rank $rank \
#           /root/paddlejob/workspace/env_run/output/zkk/2025_05_28_fefactor/baidu/paddle_internal/FastDeploy/scripts/predict_generation.py \
#           --model_name_or_path ${model_path} \
#           --input_file "./data/query-answers-list.jsonl1" \
#           --output_file ./predict_out.json \
#           --predict_model_type "W8A16C16" \
#           --dtype bfloat16 \
#           --data_format "sft" \
#           --append_bos_token "False" \
#           --min_dec_len 1 \
#           --max_dec_len 100 \
#           --batch_size $bs\
#           --top_p 0 \
#           --moe_quant_type "weight_only_int4" \
#           --use_ep "False"
# done

# exit 0

export FLAGS_use_append_attn=1
# # 使用efficien_llm whl包方式
export INFERENCE_MSG_QUEUE_ID="123321" # 随便给个名字
export FD_LOG_DIR="./log_${INFERENCE_MSG_QUEUE_ID}" #不同实例的log路径
export FD_MODEL_NAME="eb45t"

#export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/output/zkk/erniebot-dev/2024-10-25moe/miniconda3/envs/vllm_py310/lib/python3.10/site-packages/nvidia/cuda_cupti/lib:$LD_LIBRARY_PATH



/root/paddlejob/workspace/env_run/output/zkk/nsys/bin/nsys profile -c 'cudaProfilerApi' -t "cuda" -f true -o kangkang \
python /root/paddlejob/workspace/env_run/output/zkk/2025_05_28_fefactor/baidu/paddle_internal/FastDeploy/fastdeploy/demo/offline_demo.py
