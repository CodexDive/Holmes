#!/bin/bash

# Runs the "340M" parameter model (Bert - Large)

export CUDA_VISIBLE_DEVICES="5,6"
GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9999}
NUM_NODES=${1:-1}
NODE_RANK=${2:-0}
NODE_TYPE=klx
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))
export CUDA_DEVICE_MAX_CONNECTIONS=1
#CHECKPOINT_PATH=$1 #<Specify path>
#TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=$3 #<Specify path to file>/bert-vocab.json
#DATA_PATH=${DATA:-"/workspace/datasets/oscar-en-0-megatron-sub100b"}/oscar-en_text_sentence
#DATA_PATH=${DATA:-"/workspace/datasets/pile-llama"}/pile-llama_text_document
DATA_PATH=${DATA:-"/workspace/datasets/oscar-en-10k"}/oscar-en_text_sentence
TOKENIZER_PATH=${TOKEN:-"/workspace/tokenizer"}
echo "TOKENIZER_PATH: $TOKENIZER_PATH"
export LD_LIBRARY_PATH=/workspace/tools/xre-ubuntu_2004-x86_64-0.0.0.1-2024-04-26-00-05-07-daily/so:/workspace/tools/xccl_rdma-ubuntu_x86_64/so:$LD_LIBRARY_PATH
############################################ Parameters Configuration End                  ############################################


############################################ Kernel Launch Mode Configuration Begin        ############################################
#imode 模式需要同时开启下列3个环境变量
export XPU_FORCE_USERMODE_LAUNCH=1 #强制使用纯用户态高性能launch/it模式，可以获得理论最小的launch开销，需要kunlun.ko insmod时使用
export CUDART_DUMMY_REGISTER=1 #强制__cudaRegister***返回成功，用绕过pytorch混用NV官方cublas/cudnn造成的奇怪行为
unset XPU_DUMMY_EVENT
############################################ Kernel Launch Mode Configuration End          ############################################

#################################
# 14B 模型训练
#################################
XFLAGS --disable megatron_23_05
XFLAGS --enable megatron_aiak
XFLAGS --enable transformer_engine

############################################ Communication Library Configuration Begin     ############################################
# export NCCL_DEBUG=INFO # for debug
# export NCCL_DEBUG_SUBSYS=ALL
# export BKCL_DEBUG=1
export BKCL_KL3_TURBO_MODE=1
export BKCL_RING_BUFFER_SIZE=2097152

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export ALLREDUCE_ASYNC=false
export ALLGATHER_ASYNC=false
export ALLREDUCE_FUSION=0
export BKCL_TIMEOUT=360000

export BKCL_TRANS_UNSUPPORTED_DATATYPE=1
export BKCL_CCIX_RING=1
export BKCL_TREE_THRESHOLD=1
export BKCL_CCIX_BUFFER_GM=1

export XPU_ZEBU_MODE=1 # 影响ccix_inter.txt文件读取
export BKCL_XLINK_D2D=0
export BKCL_XLINK_C2C=1
export BKCL_XLINK_ETH=0

export BKCL_RING_BUFFER_GM=1
export BKCL_FORCE_SYNC=1
unset BKCL_KL3_SYSCON_FLAG
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
export DIST_MULTI_STREAM=false
############################################ Other End                                     ############################################


ulimit -c 0

HETERO_ARGS=(
    --hetero-mode pp \
    --hetero-current-device-type $NODE_TYPE \
    --hetero-device-types klx t4 \
    --hetero-pipeline-stages 1 8 1 24 \
)


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

LLAMA_MODEL_ARGS=(
    --num-layers 32 # 32
    --hidden-size 1024 # 4096
    --ffn-hidden-size 11008 # 11008
    --num-attention-heads 8 # 32
    --seq-length 512
    --max-position-embeddings 512
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 128
    --train-iters 5
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
    --use-mcore-models
    --tokenizer-model $TOKENIZER_PATH/tokenizer.model
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


MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 2
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --distributed-backend nccl
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
    --eval-iters 10
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_llama.py \
    ${LLAMA_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${HETERO_ARGS[@]} \
