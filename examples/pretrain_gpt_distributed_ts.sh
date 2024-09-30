#!/bin/bash

# Runs the "345M" parameter model


#------------------------------------------
export PYTHONPATH=/wangfangyu/test/Megatron-LM-ts:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
#export NCCL_DEBUG=INFO
export NCCL_NET_PLUGIN=none
export OMP_NUM_THREADS=4
export NCCL_NET_SHARED_BUFFERS=0

# mix with origin NV NCCL config
export NCCL_ALGO=Ring
export NCCL_P2P_NET_CHUNKSIZE=1048576
export NCCL_CHUNK_SIZE=1048576
export NCCL_BUFFSIZE=8388608
export NCCL_MAX_NCHANNELS=8
export NCCL_MIN_NCHANNELS=8
export NCCL_MAX_P2P_NCHANNELS=1
export NCCL_PROTO=Simple
export NCCL_P2P_LL_THRESHOLD=0
export IXCCL_MIX_NV=1
export IXCCL_FUSED_ENABLE=0

# IB Config
export NCCL_IB_GID_INDEX=3

export NCCL_SOCKET_IFNAME=ens2f1np1
#export NCCL_SOCKET_IFNAME=ens3f1np1
export NCCL_NET=IB

export GLOO_SOCKET_IFNAME=ens2f1np1
#export GLOO_SOCKET_IFNAME=ens3f1np1
export CUDA_DEVICES_ORDER=PCI_BUS_ID

#------------------------------------------
#export NCCL_SOCKET_IFNAME=ens3f1np1
#export GLOO_SOCKET_IFNAME=ens3f1np1
#export CUDA_DEVICE_MAX_CONNECTIONS=1
#export NCCL_NET_SHARED_BUFFERS=0
#export NCCL_DEBUG=TRACE
#export NCCL_ALGO=Ring
#export OMP_NUM_THREADS=4
#export ENABLE_FLASH_ATTENTION_WITH_IXDNN=1
#export NCCL_USE_DIRECT=1
#----------------------------------------------

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=10.107.204.37
MASTER_PORT=6003
NNODES=2
NODE_RANK=0
NODE_TYPE=ts
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=./checkpoint
#VOCAB_FILE=./data/gpt2-vocab.json
#MERGE_FILE=./data/gpt2-merges.txt
#DATA_PATH=./data/my-gpt2_text_document
DATA_PATH=./data/c4_text_document_text_document/c4_text_document_text_document
TOKENIZER_PATH=./data/c4_text_document_text_document/tokenizer.model

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
HETERO_ARGS="
    --hetero-mode pp \
    --hetero-current-device-type $NODE_TYPE \
    --hetero-device-types ts a100 \
    --hetero-pipeline-stages 1 6 1 6 \
"

GPT_ARGS="
    --num-layers 12 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --pipeline-model-parallel-size 2 \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --lr 0.00015 \
    --train-iters 50 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --attention-softmax-in-fp32 \
    --timing-log-level 2 \
    --timing-log-option all \
    --log-throughput \
"

#DATA_ARGS="
#    --data-path $DATA_PATH \
#    --vocab-file $VOCAB_FILE \
#    --merge-file $MERGE_FILE \
#    --split 949,50,1
#"

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --split 1 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_PATH} \
"

OUTPUT_ARGS="
    --log-interval 1 \
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
