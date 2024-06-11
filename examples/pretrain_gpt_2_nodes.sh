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
NNODES=2
NODE_RANK=$1
NODE_DEVICES=8
NODE_TYPE=$2

CHECKPOINT_PATH=checkpoint
TENSORBOARD_PATH=tensorboard
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=/gpfs/gpt3/code/Holmes/local_data/huggingface_text_document

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
    --hetero-pipeline-stages 1 16 1 14 \
    --use-hetnet
"

GPT_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --num-layers 30 \
    --hidden-size 3072 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 512 \
    --lr 0.8e-4 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 0.8e-5 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --use-rotary-position-embeddings \
    --use-flash-attn \
    --no-gradient-accumulation-fusion \
    --recompute-activations \
    --swiglu \
    --overlapped-distributed-optimizer \
    --use-checkpoint-opt_param-scheduler \
    --use-hetnet \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --tensorboard-dir $TENSORBOARD_PATH \
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
