#!/bin/bash

# Runs the "340M" parameter model (Bert - Large)

export CUDA_VISIBLE_DEVICES="4,5,6,7"
GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9967}
NUM_NODES=${1:-1}
NODE_RANK=${2:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))
export CUDA_DEVICE_MAX_CONNECTIONS=1
#CHECKPOINT_PATH=$1 #<Specify path>
#TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=$3 #<Specify path to file>/bert-vocab.json
#DATA_PATH=${DATA:-"/workspace/datasets/oscar-en-0-megatron-sub100b"}/oscar-en_text_sentence
#DATA_PATH=${DATA:-"/workspace/datasets/pile-llama"}/pile-llama_text_document
DATA_PATH=${DATA:-"/Holmes/data/oscar/my-llama_text_document"}
TOKENIZER_PATH=${TOKEN:-"/Holmes/data/"}


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

LLAMA_MODEL_ARGS=(
    --num-layers 16 # 32
    --hidden-size 1024 # 4096
    --ffn-hidden-size 11008 # 11008
    --num-attention-heads 32 # 32
    --seq-length 4096
    --max-position-embeddings 4096
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
    ${EVAL_AND_LOGGING_ARGS[@]}
