#!/bin/bash

# Runs Mixtral 8x7B model on 16 A100 GPUs
export MACA_PATH=/opt/maca
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export MACA_CLANG=${MACA_PATH}/mxgpu_llvm
export DEVINFO_ROOT=${MACA_PATH}
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export CUDA_PATH=${CUCC_PATH}
export PATH=${CUCC_PATH}:${MACA_PATH}/bin:${MACA_CLANG}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=$PYTHONPATH:/root/FlagScale
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MALLOC_THRESHOLD=99
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"
export MACA_SMALL_PAGESIZE_ENABLE=1
export MALLOC_THRESHOLD=95
export MCPYTORCH_DISABLE_PRINT=1

export MCCL_NET_GDR_LEVEL=7
#ENV MCCL_MIN_NCHANNELS=16
export MCCL_MAX_NCHANNELS=16
export MCCL_P2P_LEVEL=SYS
export MCCL_LIMIT_RING_LL_THREADTHRESHOLDS=1
export FORCE_ACTIVATE_WAIT=1

export MHA_USE_BLAS=ON
export MHA_BWD_NO_ATOMIC_F64=1
export SET_DEVICE_NUMA_PREFERRED=1

export MAX_JOBS=20
export PYTORCH_ENABLE_SAME_RAND_A100=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/ai_mirror/sfwang/llama2-70B/log
TOKENIZER_MODEL=/ai_mirror/sfwang/llama2-70B/tokenizer/tokenizer.model
DATA_PATH=/ai_mirror/sfwang/llama2-70B/data/llama_00_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --num-layers 1 \
    --hidden-size 5120 \
    --ffn-hidden-size 17920 \
    --num-attention-heads 40 \
    --init-method-std 0.02 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --group-query-attention \
    --num-query-groups 8 \
    --no-masked-softmax-fusion \
    --no-position-embedding \
    --use-flash-attn
)

MOE_ARGS=(
    --num-experts 8 \
    --expert-model-parallel-size 2 \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 2 \
    --moe-aux-loss-coeff 1e-3 \
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --data-path $DATA_PATH \
    --split 99990,8,2
)

TRAINING_ARGS=(
    --micro-batch-size 1 \
    --global-batch-size 128 \
    --lr 1e-4 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 0.1 \
    --lr-warmup-iters 500 \
    --clip-grad 1.0 \
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --use-distributed-optimizer
)

LOGGING_ARGS=(
    --log-interval 1 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --no-load-optim \
    --no-load-rng
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Mixtral-Finetuning" } \
        --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B"} 
    )
fi

torchrun ${DISTRIBUTED_ARGS[@]} ../pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
