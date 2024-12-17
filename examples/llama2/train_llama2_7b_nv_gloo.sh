#!/bin/bash
# Runs the "340M" parameter model (Bert - Large)

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`
# 10.107.204.3: ens20f0np0  10.107.206.60: ens5f0np0
export GLOO_SOCKET_IFNAME=ens24np0
#export UCX_NET_DEVICES=mlx5_cx4lx_3
export UCX_NET_DEVICES=ens24np0
#export UCX_LOG_LEVEL=DEBUG

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"10.107.204.72"}
MASTER_PORT=${MASTER_PORT:-9997}
NUM_NODES=${1:-2}
NODE_RANK=${2:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))
export CUDA_DEVICE_MAX_CONNECTIONS=1
export BKCL_DEBUG=0
CHECKPOINT_PATH="/Megatron-LM-mixed/checkpoint"
TENSORBOARD_LOGS_PATH="Megatron-LM-mixed/tensorboard"
VOCAB_FILE=$3 #<Specify path to file>/bert-vocab.json
#DATA_PATH=${DATA:-"/workspace/datasets/oscar-en-0-megatron-sub100b"}/oscar-en_text_sentence
#DATA_PATH=${DATA:-"/workspace/datasets/pile-llama"}/pile-llama_text_document
DATA_PATH=${DATA:-"/Megatron-LM-mixed/data/oscar/my-llama_text_document"}
TOKENIZER_PATH=${TOKEN:-"/Megatron-LM-mixed/data"}
echo "TOKENIZER_PATH: $TOKENIZER_PATH"
# export LD_LIBRARY_PATH=/workspace/tools/xre-ubuntu_2004-x86_64-0.0.0.1-2024-04-26-00-05-07-daily/so:/workspace/tools/xccl_rdma-ubuntu_x86_64/so:$LD_LIBRARY_PATH
############################################ Parameters Configuration End                  ############################################


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

LLAMA_MODEL_ARGS=(
    --num-layers 16 # 32
    --hidden-size 2048 # 4096
    --ffn-hidden-size 11008 # 11008
    --num-attention-heads 32 # 32
    --seq-length 4096 # 4096
    --max-position-embeddings 4096 #4096
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1024 #1024
    --train-iters 20000
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
    --tokenizer-model $TOKENIZER_PATH/tokenizer.model
    --tokenizer-type Llama2Tokenizer
    --normalization LayerNorm
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
    --fp16
)

#   --use-distributed-optimizer
#   --overlap-param-gather
MODEL_PARALLEL_ARGS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 2
  --use-distributed-optimizer
  --distributed-backend gloo
  --local-distributed-backend nccl
  --cross-distributed-backend gloo
#  --overlap-grad-reduce # bug: param.grad being None is not safe when overlap_grad_reduce is True
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --vocab-file null
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10
    --eval-interval 10
    --eval-iters 5
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
)

EXTRA_ARGS=(
    --transformer-impl local
    --distributed-timeout-minutes 10
)

echo "torchrun ${DISTRIBUTED_ARGS[@]} pretrain_llama.py ${LLAMA_MODEL_ARGS[@]} ${TRAINING_ARGS[@]} ${MODEL_PARALLEL_ARGS[@]} ${DATA_ARGS[@]} ${EVAL_AND_LOGGING_ARGS[@]}"

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_llama.py \
    ${LLAMA_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${EXTRA_ARGS[@]}


