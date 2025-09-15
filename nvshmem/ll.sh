
source /root/paddlejob/workspace/env_run/output/zkk/erniebot-dev/2024-10-25moe/miniconda3/bin/activate  /root/paddlejob/workspace/env_run/output/zkk/erniebot-dev/2024-10-25moe/miniconda3/envs/deepseekv3_310_new
export LD_LIBRARY_PATH=/usr/local/nccl/:$LD_LIBRARY_PATH
# 屏蔽平台预设的环境变量，因为框架采用兼容升级，检测到这些配置会使用原方式启动
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
#source /root/paddlejob/workspace/env_run/output/zkk/2025_03_28_45T/wuliji.sh
nnodes=$PADDLE_TRAINERS_NUM
rank=$PADDLE_TRAINER_ID

for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
  unset ${name}
done

START_RANK=0
END_RANK=$nnodes
END_RANK=1

if [[ $rank -lt $START_RANK ]]; then
    echo "rank $rank exit"
    exit 0
fi

if [[ $rank -ge $END_RANK ]]; then
    echo "rank $rank exit"
    exit 0
fi


rank=$(($rank-$START_RANK))
nnodes=$(($END_RANK-$START_RANK))
master=`cat /root/paddlejob/workspace/hostfile | head -n $(($START_RANK+1)) | tail -n 1 | awk '{print $1}'`
port=36677



export device="0,1,2,3,4,5,6,7"

export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=xgbe0
export NCCL_DEBUG=info
export LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/output/zkk/lidong/nvshmem/lib:$LD_LIBRARY_PATH



echo $nnodes
echo $rank

#/root/paddlejob/workspace/env_run/output/zkk/nsys/bin/nsys profile -c 'cudaProfilerApi' -t "cuda" -f true -o kangkang \
python -m paddle.distributed.launch --log_dir output/paddle_distributed_logs --gpus ${device} --master $master:$port --nnodes $nnodes --rank $rank \
/root/paddlejob/workspace/env_run/output/zkk/HomeWork/test_ll.py
#/root/paddlejob/workspace/env_run/output/zkk/erniebot-dev/2024-10-25moe/Paddle/test/collective/test_m2n.py





#python -m paddle.distributed.launch --gpus ${device} /root/paddlejob/workspace/env_run/output/zkk/HomeWork/A.py

