#!/bin/bash

# Runs the "340M" parameter model (Bert - Large)
set -x
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="ens17f0np0"
export GLOO_SOCKET_IFNAME="ens17f0np0"
export NCCL_IB_HCA=mlx5_cx6_0,mlx5_cx6_1,mlx5_cx6_2,mlx5_cx6_3
export CUDA_DEVICE_MAX_CONNECTIONS=1


export UCX_NET_DEVICES=mlx5_cx6_0:1
# export ZCCL_LOG_LEVEL=debug

# export UCX_WARN_UNUSED_ENV_VARS=n
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"10.107.204.72"}
MASTER_PORT=${MASTER_PORT:-6789}
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
	--pipeline-model-parallel-size 2
    --use-distributed-optimizer
    --overlap-grad-reduce
    # --distributed-backend nccl
    #--sequence-parallel
    --distributed-backend gloo
    --local-distributed-backend nccl
    --cross-distributed-backend zccl
    --use-gdr
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
    2>&1 | tee nv_log.txt

