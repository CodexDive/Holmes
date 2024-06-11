#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO

export NCCL_SOCKET_IFNAME=band0
export GLOO_SOCKET_IFNAME=band0

# Change for multinode config
MASTER_ADDR=<Specify Your Master Addr>
MASTER_PORT=<Specify Your Master Addr>
HOSTNAME=$(hostname)
NNODES=4
NODE_RANK=$1
NODE_DEVICES=8
NODE_TYPE=$2

CHECKPOINT_PATH=./checkpoint
VOCAB_FILE=./data/gpt2-vocab.json
MERGE_FILE=./data/gpt2-merges.txt
DATA_PATH=./data/my-gpt2_text_document


DISTRIBUTED_ARGS="
    --nproc_per_node $NODE_DEVICES \
    --nnodes $NNODES \
    --node_rank $((NODE_RANK+1)) \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


HETERO_ARGS="
    --hetero-mode pp \
    --hetero-current-device-type $NODE_TYPE \
    --hetero-device-types ib roce \
    --hetero-pipeline-stages 1 26 1 22 \
    --use-hetnet
"
  

GPT_ARGS="
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 2 \
    --sequence-parallel \
    --num-layers 48 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 1536 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --use-rotary-position-embeddings \
    --use-flash-attn \
    --no-gradient-accumulation-fusion \
    --recompute-activations \
    --swiglu \
    --use-distributed-optimizer \
    --use-checkpoint-opt_param-scheduler \
    --overlapped-distributed-optimizer \
"
  
DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 10 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"
  
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $HETERO_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
