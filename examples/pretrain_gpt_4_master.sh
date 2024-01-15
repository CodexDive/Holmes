#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_IBEXT_DISABLE=1
# export NCCL_NET=Socket
# export NCCL_IB_HCA=mlx5_4,mlx5_5,mlx5_8,mlx5_9

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=<Specify Your Master Addr>
MASTER_PORT=<Specify Your Master Port>
NNODES=4
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


CHECKPOINT_PATH=checkpoint
TENSORBOARD_PATH=tensorboard
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=/gpfs/gpt3/code/Holmes/local_data/huggingface_text_document
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

GPT_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 2 \
    --sequence-parallel \
    --num-layers 36 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 2048 \
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
    --use-distributed-optimizer \
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
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
