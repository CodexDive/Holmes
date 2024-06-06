#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,3,5
export NCCL_SOCKET_IFNAME=ens11
export GLOO_SOCKET_IFNAME=ens11
# Change for multinode config
MASTER_ADDR=10.11.12.218
MASTER_PORT=6000
NNODES=2
NODE_RANK=1
NODE_DEVICES=2
NODE_TYPE=a100

CHECKPOINT_PATH=./checkpoint
VOCAB_FILE=./data/gpt2-vocab.json
MERGE_FILE=./data/gpt2-merges.txt
DATA_PATH=./data/my-gpt2_text_document


DISTRIBUTED_ARGS="
    --nproc_per_node $NODE_DEVICES \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


# HETERO_ARGS="
#     --hetero-mode pp \
#     --hetero-current-device-type $NODE_TYPE \
#     --hetero-device-types t4 a100 \
#     --hetero-pipeline-stages 1 4 1 20 \
#     --use-hetnet \
# "
#  

HETERO_ARGS="
    --hetero-mode dp \
    --hetero-current-device-type $NODE_TYPE \
    --hetero-device-types t4 a100 \
    --hetero-micro-batch-sizes 8 1 2 2 \
    --use-hetnet
"

# --global-batch-size 12 \
# --micro-batch-size 1 \
GPT_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --pipeline-model-parallel-size 1 \
    --global-batch-size 12 \
    --lr 0.00015 \
    --train-iters 50 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --timing-log-level 2 \
    --log-throughput \
    --empty-unused-memory-level 0 \
    --use-checkpoint-opt_param-scheduler \
    --timing-log-level 2 \
    --attention-softmax-in-fp32 \
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
