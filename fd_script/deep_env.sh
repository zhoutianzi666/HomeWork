export LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/output/zkk/lidong/nvshmem/lib:$LD_LIBRARY_PATH

export $(bash /root/paddlejob/workspace/env_run/output/zkk/HomeWork/rdma.sh gpu)
export $(bash /root/paddlejob/workspace/env_run/output/zkk/HomeWork/rdma.sh cpu)


export NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${KV_CACHE_SOCKET_IFNAME}


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


