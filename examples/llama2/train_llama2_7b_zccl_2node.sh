#! /bin/bash

set -e
set -x

source /workspace/zjlab_llama2_multi_node_env.sh

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

#################################
# 清理环境
#################################
pkill -9 python || true

#################################
# 数据集部分
#################################
# Define VAR with default values if they are not already set
DATA_PATH=${DATA_PATH:-"/workspace/datasets/oscar-en-10k/oscar-en_text_sentence"}
# CHECKPOINT_LOAD_PATH=${CHECKPOINT_LOAD_PATH:-"/mnt/data1/megatron_datasets/llama/checkpoints/megatron_llama_13b_checkpoint_tp8_pp1_fp32"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/tokenizer/tokenizer.model"}
ITERATION=${ITERATION:-300}

# 需要根据实际情况打开
BKCL_SO=${BKCL_SO:-"/workspace/tools/xccl_rdma-ubuntu_x86_64/so"}
RUNTIME_SO=${RUNTIME_SO:-"/workspace/tools/xre-ubuntu_2004-x86_64-0.0.0.1-2024-04-26-00-05-07-daily/so"}
export LD_LIBRARY_PATH=$BKCL_SO:$RUNTIME_SO:$LD_LIBRARY_PATH
#export ZCCL_LOG_LEVEL=INFO
#export ZCCL_COLL_TRACE=DEBUG
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
#MASTER_ADDR=megatron-zccl-pt-triton-02-master-0.megatron-zccl-pt-triton-02
MASTER_PORT=${MASTER_PORT:-9999}
NNODES=${NNODES:-2}
# NODE_RANK按需修改
NODE_RANK=$1

ulimit -c 0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"



LLAMA_MODEL_ARGS=(
      --num-layers  32
      --hidden-size  4096
      --ffn-hidden-size  11008
      --num-attention-heads  32
      --seq-length  4096
      --max-position-embeddings 4096
)
# --fp16
TRAINING_ARGS=(
    --fp16
    --micro-batch-size 4
    --global-batch-size 4
    --train-iters 1000
    --weight-decay 1e-2
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --lr 1.5e-4
    --lr-decay-iters 10251
    --lr-decay-style cosine
    --min-lr 1.5e-5
    --weight-decay 1e-2
    --lr-warmup-iters 2000
    --clip-grad 1.0
    --swiglu
    --use-flash-attn
    --use-mcore-models
    --tokenizer-model $TOKENIZER_PATH
    --tokenizer-type Llama2Tokenizer
    --normalization RMSNorm
    --disable-bias-linear
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --initial-loss-scale 64
    --position-embedding-type rope
    --make-vocab-size-divisible-by 128
    --untie-embeddings-and-output-weights
    --norm-epsilon 1e-6
    --optimizer adam
    #--no-bias-swiglu-fusion
    --seed 1234
    --attention-dropout 0
    --hidden-dropout 0
    --log-throughput
)

#   --use-distributed-optimizer
#   --overlap-param-gather
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 16
    --use-distributed-optimizer
    --overlap-grad-reduce
    --distributed-backend zccl
    #--sequence-parallel
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --vocab-file null
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 1000
    --eval-interval 1000
    --eval-iters 2
)


#OUTPUT_ARGS="--log-interval 1 \
#             --save-interval 10000 \
#             --eval-interval 1000 \
#             --eval-iters 5"

#OTHER_ARGS="--data-path $DATA_PATH \
#            --data-impl mmap \
#            --split 949,50,1 \
#            --distributed-backend nccl"

#################################
# driver方式部分
#################################
#pmode 模式需要开启下面环境变量
export CUDART_DUMMY_REGISTER=1 #强制__cudaRegister***返回成功，用绕过pytorch混用NV官方cublas/cudnn造成的奇怪行为

#imode 模式需要同时开启下列3个环境变量
export CUDART_DUMMY_REGISTER=1 #强制__cudaRegister***返回成功，用绕过pytorch混用NV官方cublas/cudnn造成的奇怪行为
export XPU_FORCE_USERMODE_LAUNCH=1 #强制使用纯用户态高性能launch/it模式，可以获得理论最小的launch开销，需要kunlun.ko insmod时使用
# export XPU_DUMMY_EVENT=1 #强制event相关API返回成功，当前用于event还未ready的纯用户态高性能launch/wait模式
# 如果要开启多流 则需要 关掉 XPU_DUMMY_EVENT
export DIST_MULTI_STREAM=true
#################################
# 算子部分
#################################
export XMLIR_FA_GEMM_TYPE=float16 # FA GEMM type默认为tf32，较慢，需要指定为fp16
#################################
# 算子检查部分
#################################
export XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL=0 # 禁止XDNN出错后fallback CPU
export XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL=False #禁止任何fallback
export XMLIR_DUMP_FALLBACK_OP_LIST_BOOL=true

#################################
# 通信部分设置
#################################
# 需要视机器情况来做修改(命令 ip a查看)
#export BKCL_RDMA_NICS=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4 
# export BKCL_RDMA_NICS=xgbe1,xgbe1,xgbe2,xgbe2,xgbe3,xgbe3,xgbe4,xgbe4

# rdma 需要设置
export BKCL_FORCE_L3_RDMA=0
export BKCL_ENABLE_XDR=1  
export BKCL_RDMA_FORCE_TREE=1 
export BKCL_TREE_THRESHOLD=1
export ZCCL_LOG_LEVEL=INFO

#echo "whosyourdaddy" >> /proc/kunlun/dev0/info
#echo "whosyourdaddy" >> /proc/kunlun/dev1/info
#echo "whosyourdaddy" >> /proc/kunlun/dev2/info
#echo "whosyourdaddy" >> /proc/kunlun/dev3/info
#echo "whosyourdaddy" >> /proc/kunlun/dev4/info
#echo "whosyourdaddy" >> /proc/kunlun/dev5/info
#echo "whosyourdaddy" >> /proc/kunlun/dev6/info
#echo "whosyourdaddy" >> /proc/kunlun/dev7/info
#echo "3600000" > /proc/kunlun/dev0/task_timeout_detect_threshold_in_ms
#echo "3600000" > /proc/kunlun/dev1/task_timeout_detect_threshold_in_ms
#echo "3600000" > /proc/kunlun/dev2/task_timeout_detect_threshold_in_ms
#echo "3600000" > /proc/kunlun/dev3/task_timeout_detect_threshold_in_ms
#echo "3600000" > /proc/kunlun/dev4/task_timeout_detect_threshold_in_ms
#echo "3600000" > /proc/kunlun/dev5/task_timeout_detect_threshold_in_ms
#echo "3600000" > /proc/kunlun/dev6/task_timeout_detect_threshold_in_ms
#echo "3600000" > /proc/kunlun/dev7/task_timeout_detect_threshold_in_ms

# 检查 /dev/xdrdrv 是否存在
if [ ! -e "/dev/xdrdrv" ]; then
    echo "Error: /dev/xdrdrv not found! PLEASE CHECK YOUR DOCKER CONTAINER"
    exit 1
fi

#################################
# BKCL C2C部分
#################################
export BKCL_CCIX_RING=1
export BKCL_TREE_THRESHOLD=1
export BKCL_CCIX_BUFFER_GM=1

# ccix_inner_8chips
cat > ccix_inter.txt <<EOF
[chip 0, port 0] <===> [chip 6, port 3]
[chip 4, port 1] <===> [chip 5, port 2]
[chip 2, port 3] <===> [chip 4, port 0]
[chip 2, port 1] <===> [chip 3, port 2]
[chip 1, port 1] <===> [chip 3, port 1]
[chip 0, port 1] <===> [chip 1, port 2]
[chip 0, port 2] <===> [chip 2, port 2]
[chip 0, port 3] <===> [chip 3, port 3]
[chip 1, port 0] <===> [chip 2, port 0]
[chip 1, port 3] <===> [chip 7, port 0]
[chip 5, port 1] <===> [chip 7, port 1]
[chip 3, port 0] <===> [chip 5, port 3]
[chip 4, port 3] <===> [chip 7, port 3]
[chip 4, port 2] <===> [chip 6, port 2]
[chip 6, port 1] <===> [chip 7, port 2]
[chip 5, port 0] <===> [chip 6, port 0]
EOF


export XPU_ZEBU_MODE=1 # 影响ccix_inter.txt文件读取
export BKCL_XLINK_D2D=0
export BKCL_XLINK_C2C=1
export BKCL_XLINK_ETH=0
export BKCL_TRANS_UNSUPPORTED_DATATYPE=1
export BKCL_RING_BUFFER_GM=1
export BKCL_FORCE_SYNC=1
export BKCL_KL3_TURBO_MODE=1 
export BKCL_RING_BUFFER_SIZE=2097152
#export XPUSIM_TOPOLOGY_FILE="ccix_inter.txt"
export ALLREDUCE_ASYNC=false
export ALLGATHER_ASYNC=false
export ALLREDUCE_FUSION=0
export BKCL_TIMEOUT=360000
export CUDA_DEVICE_MAX_CONNECTIONS=1
unset BKCL_KL3_SYSCON_FLAG
#################################
# DEBUG 用途部分
#################################
# export XLOG_LEVEL="capture=info,dist=info"
# export XPURT_DISPATCH_MODE=PROFILING
# export XPUAPI_DEBUG=0x1001
#export PYTORCH_NO_CUDA_MEMORY_CACHING=1
#export BKCL_PERF_DEBUG=1
#export BKCL_DEBUG=1
#export BKCL_DUMP=1
#export BKCL_DEBUG_SKIP_RUN=1
#export CUDA_DEVICE_MAX_CONNECTIONS=1


#################################
# XFLAGS插件设置
#################################
XFLAGS --list
XFLAGS --enable megatron_aiak
XFLAGS --disable megatron_23_05

#################################
# 13B 模型训练
#################################
PYTHONPATH="$ROOT_DIR/..":$PYTHONPATH torchrun $DISTRIBUTED_ARGS $ROOT_DIR/../../pretrain_llama.py \
    ${LLAMA_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
