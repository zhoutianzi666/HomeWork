
# 分别配置nvshmem, mpi, cuda, nccl的路径
export NVSHMEM_HOME=/root/paddlejob/workspace/env_run/output/zkk/lidong/nvshmem
#export NVSHMEM_HOME=/root/paddlejob/workspace/env_run/output/zkk/nvshmem/so_from_acg/nvshmem



export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.5a1/
export MPI_HOME=/usr/local/openmpi-4.1.5
export NCCL_HOME=/root/paddlejob/workspace/env_run/output/zkk/nvshmem/nccl/build/

# 将nvshmem, mpi, nccl的头文件和库文件都添加到路径中
export CPATH=$CPATH:$NVSHMEM_HOME/include:$MPI_HOME/include:$NCCL_HOME/include
export LIBRARY_PATH=$NVSHMEM_HOME/lib:$MPI_HOME/lib:$NCCL_HOME/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$MPI_HOME/lib:$NCCL_HOME/lib:$LD_LIBRARY_PATH

# export LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/output/zkk/nvshmem/gdrcopy-2.4.4:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/output/zkk/nvshmem/so_from_acg:$LD_LIBRARY_PATH


# 配置编译的其他参数
export NVCC_GENCODE="arch=compute_90,code=sm_90"
export NVSHMEM_MPI_SUPPORT=1
export NVSHMEM_USE_NCCL=1

# 编译执行



#nvcc -rdc=true -ccbin g++ -gencode=$NVCC_GENCODE atomic.cu -o test.out -lnvshmem_host -lnvshmem_device -lmpi 
#nvcc -rdc=true -ccbin g++ -gencode=$NVCC_GENCODE unique_id.cu -o test.out -lnvshmem_host -lnvshmem_device -lmpi 

export NVSHMEM_DISABLE_P2P=true
# export NVSHMEM_IB_ENABLE_IBGDA=true

#python /root/paddlejob/workspace/env_run/output/zkk/erniebot-dev/script/sync.py file ./test.out

./test.out
#/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun --allow-run-as-root --host "127.0.0.1:8,10.94.130.151" ./test.out

