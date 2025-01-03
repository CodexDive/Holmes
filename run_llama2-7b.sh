#!/bin/bash

# Runs the "340M" parameter model (Bert - Large)
set -x
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`
export GLOO_SOCKET_IFNAME=ens20f0np0
#export UCX_NET_DEVICES=mlx5_cx4lx_3
export UCX_NET_DEVICES=ens20f0np0
#export UCX_LOG_LEVEL=DEBUG
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"10.107.204.3"}
MASTER_PORT=${MASTER_PORT:-4567}
NUM_NODES=${1:-2}
NODE_RANK=${2:-0}
NODE_TYPE=klx
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))
export CUDA_DEVICE_MAX_CONNECTIONS=1
#CHECKPOINT_PATH=$1 #<Specify path>
#TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=$3 #<Specify path to file>/bert-vocab.json
#DATA_PATH=${DATA:-"/workspace/datasets/oscar-en-0-megatron-sub100b"}/oscar-en_text_sentence
#DATA_PATH=${DATA:-"/workspace/datasets/pile-llama"}/pile-llama_text_document
DATA_PATH=${DATA:-"/Holmes/data/oscar/my-llama_text_document"}
TOKENIZER_PATH=${TOKEN:-"/Holmes/data"}
echo "TOKENIZER_PATH: $TOKENIZER_PATH"
export LD_LIBRARY_PATH=/workspace/tools/xre-ubuntu_2004-x86_64-0.0.0.1-2024-04-26-00-05-07-daily/so:/workspace/tools/xccl_rdma-ubuntu_x86_64/so:$LD_LIBRARY_PATH
############################################ Parameters Configuration End                  ############################################


############################################ Kernel Launch Mode Configuration Begin        ############################################
#imode 模式需要同时开启下列3个环境变量
# export XPU_FORCE_USERMODE_LAUNCH=1 #强制使用纯用户态高性能launch/it模式，可以获得理论最小的launch开销，需要kunlun.ko insmod时使用
export CUDART_DUMMY_REGISTER=1 #强制__cudaRegister***返回成功，用绕过pytorch混用NV官方cublas/cudnn造成的奇怪行为
unset XPU_DUMMY_EVENT
############################################ Kernel Launch Mode Configuration End          ############################################

#################################
# 14B 模型训练
#################################
XFLAGS --disable megatron_23_05
XFLAGS --disable megatron_aiak
XFLAGS --enable megatron_core_0_6
XFLAGS --enable transformer_engine

############################################ Communication Library Configuration Begin     ############################################
# export NCCL_DEBUG=INFO # for debug
# export NCCL_DEBUG_SUBSYS=ALL
# export BKCL_DEBUG=1
# export BKCL_KL3_TURBO_MODE=1
# export BKCL_RING_BUFFER_SIZE=2097152

# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_HCA=mlx5
# export NCCL_IB_GID_INDEX=3
# export ALLREDUCE_ASYNC=false
# export ALLGATHER_ASYNC=false
# export ALLREDUCE_FUSION=0
export BKCL_TIMEOUT=360000

# export BKCL_TRANS_UNSUPPORTED_DATATYPE=1
# export BKCL_CCIX_RING=1
# export BKCL_TREE_THRESHOLD=1
# export BKCL_CCIX_BUFFER_GM=1

# export XPU_ZEBU_MODE=1 # 影响ccix_inter.txt文件读取
# export BKCL_XLINK_D2D=0
# export BKCL_XLINK_C2C=1
# export BKCL_XLINK_ETH=0

# export BKCL_RING_BUFFER_GM=1
# export BKCL_FORCE_SYNC=1
# unset BKCL_KL3_SYSCON_FLAG
############################################ Communication Library Configuration End       ############################################


############################################ Op Performance Configuration Begin            ############################################
# Op performance env
#export XPUAPI_DEFAULT_SIZE=4000000000
export XBLAS_FC_HBM_VERSION=40 # 需要指定HBM版
export XPU_FORCE_CODE_PARAM_LOCATE_IN_L3=1 #高频hbm必开, stable/low选开
export XMLIR_FA_GEMM_TYPE=float16 #FA GEMM type默认为tf32，较慢，需要指定为fp16
export FAST_SWIGLU_ENABLE=1 # 查表SWIGLU算子实现
############################################ Op Performance Configuration End              ############################################


############################################ XMLIR Configuration Begin                     ############################################
# Prohibit op fallback to cpu
export XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL=0 # 禁止XDNN出错后fallback CPU
export XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL=False #禁止任何fallback
############################################ XMLIR Configuration End                       ############################################


############################################ Other Begin                                   ############################################
# profile xdnn_pytorch
# export XPURT_DISPATCH_MODE=PROFILING
#For Multi-Stream
export DIST_MULTI_STREAM=true
############################################ Other End                                     ############################################


ulimit -c 0

#HETERO_ARGS=(
#    --hetero-mode pp \
#    --hetero-current-device-type $NODE_TYPE \
#    --hetero-device-types klx t4 \
#    --hetero-pipeline-stages 1 8 1 24 \
#)


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --node_rank $NODE_RANK
)

LLAMA_MODEL_ARGS=(
    --num-layers 32 # 32
    --hidden-size 4096 # 4096
    --ffn-hidden-size 11008 # 11008
    --num-attention-heads 32 # 32
    --seq-length 4096
    --max-position-embeddings 4096
)


TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1024
    --train-iters 20
    --weight-decay 1e-2
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --fp16
    --lr 1.5e-4
    --lr-decay-iters 10251
    --lr-decay-style cosine
    --min-lr 1.5e-5
    --weight-decay 1e-2
    --lr-warmup-iters 2000
    --clip-grad 1.0
    --swiglu
    --use-flash-attn
    --tokenizer-model $TOKENIZER_PATH/tokenizer.model
    --tokenizer-type Llama2Tokenizer
    --normalization RMSNorm
    --disable-bias-linear
    --attention-softmax-in-fp32
    --initial-loss-scale 64
    --make-vocab-size-divisible-by 128
    --norm-epsilon 1e-6
    --optimizer adam
    #--no-bias-swiglu-fusion
    --seed 1234
    --attention-dropout 0
    --hidden-dropout 0
    --log-throughput
    # new
    --group-query-attention
    --num-query-groups 32
    --untie-embeddings-and-output-weights
    --no-position-embedding
    --use-rotary-position-embeddings
    --max-position-embeddings 4096
    --transformer-impl local
)


MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1 
    --use-distributed-optimizer
    --overlap-grad-reduce
    --distributed-backend nccl
    #--sequence-parallel
    --distributed-backend gloo
    --local-distributed-backend xccl
    --cross-distributed-backend gloo
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
    --eval-iters 10
)

## new bkcl env
export BKCL_TREE_THRESHOLD=0
export BKCL_RING_BUFFER_SIZE=2097152
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_FORCE_TREE=1
export BKCL_RDMA_NICS=ens11np0,ens11np0,ens13np0,ens13np0,ens15np0,ens15np0,ens17np0,ens17np0
export BKCL_FORCE_L3_RDMA=0
export BKCL_RING_HOSTID_USE_RANK=1
#export XLOG_LEVEL="dist=info"
#export BKCL_DEBUG=1
#export BKCL_DUMP=1
#export SAVE_LOG_FILE_WITH_RANK_ID=false

#export BKCL_RDMA_PROXY_DISABLE=1
#export BKCL_FLAT_RING=1

# export BKCL_SOCKET_IFNAME=ens11np0,ens11np0,ens13np0,ens13np0,ens15np0,ens15np0,ens17np0,ens17np0,ens20f0np0
# export NCCL_SOCKET_IFNAME=ens11np0,ens11np0,ens13np0,ens13np0,ens15np0,ens15np0,ens17np0,ens17np0,ens20f0np0

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_llama.py \
    ${LLAMA_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${HETERO_ARGS[@]} \

