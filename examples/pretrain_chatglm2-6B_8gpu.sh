#! /bin/bash

# Setting the environment variables
export MACA_PATH=/opt/maca
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export MACA_CLANG=${MACA_PATH}/mxgpu_llvm
export DEVINFO_ROOT=${MACA_PATH}
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export CUDA_PATH=${CUCC_PATH}
export PATH=${CUCC_PATH}:${MACA_PATH}/bin:${MACA_CLANG}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"
export MACA_SMALL_PAGESIZE_ENABLE=1
export MCPYTORCH_DISABLE_PRINT=1

export MCCL_NET_GDR_LEVEL=7
#ENV MCCL_MIN_NCHANNELS=16
#export MCCL_MAX_NCHANNELS=16
export MCCL_P2P_LEVEL=SYS
export MCCL_LIMIT_RING_LL_THREADTHRESHOLDS=1
export FORCE_ACTIVATE_WAIT=1

export MHA_USE_BLAS=ON
export MHA_BWD_NO_ATOMIC_F64=1
export SET_DEVICE_NUMA_PREFERRED=1

export MAX_JOBS=20
export PYTORCH_ENABLE_SAME_RAND_A100=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Distributed training variables
NNODES=1
GPUS_PER_NODE=8
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
NODE_RANK=0
MASTER_PORT=12453
MASTER_ADDR=localhost

# Parallelism variables
TP=1
PP=2
DP=$((${GPU_NUM}/${TP}/${PP}))

# Network size variables

HIDDEN_SIZE=4096
NUM_HEAD=32
NUM_QUERY_GROUP=2
NUM_LAYERS=28
FFN_HIDDEN_SIZE=13696
NORM_EPS=1e-5

DROP_OUT=0.0
MAX_SEQ_LEN=4096
MAX_POSITION_EMBEDDINGS=4096

# Paths
BASE_PATH=./chatglm2-6B
SRC_PATH=../pretrain_chatglm.py

LOG_NAME=chatglm2-6b_pretrain_WS${WORLD_SIZE}_TP${TP}_PP${PP}
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p ${BASE_PATH}/log/${LOG_NAME}

DATA_PATH=/mnt/nvme/nvme1/sfwang/AdvertiseGen/train_summary_document
DATA_CACHE_PATH="./data_cache/${LOG_NAME}"
mkdir -p ${DATA_CACHE_PATH}

SAVE_PATH=${BASE_PATH}/checkpoint/${LOG_NAME}
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p ${BASE_PATH}/log/${LOG_NAME}

TOKENIZER_PATH=/mnt/nvme/nvme1/sfwang/AdvertiseGen/tokenizer.model

# Set training command
LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${NNODES} \
       --node_rank ${NODE_RANK} \
       --master_addr ${MASTER_ADDR} \
       --master_port ${MASTER_PORT} \
       "

#LAUNCHER=" \
#       torchrun \
#       --nproc_per_node ${GPUS_PER_NODE} \
#       --nnodes ${NNODES} \
#       --node_rank ${NODE_RANK} \
#       "


DISTRIBUTED_ARGS=" \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --distributed-backend nccl \
       --use-distributed-optimizer \
       --sequence-parallel \
       --overlap-grad-reduce
       "    

#NETWORK_SIZE_ARGS=" \
#       --num-layers ${NUM_LAYERS} \
#       --hidden-size ${HIDDEN_SIZE} \
#       --num-attention-heads ${NUM_HEAD} \
#       --group-query-attention \
#       --num-query-groups ${NUM_QUERY_GROUP} \
#       --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
#       --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
#       --norm-epsilon ${NORM_EPS} \
#       --normalization RMSNorm \
#       --swiglu \
#       --untie-embeddings-and-output-weights \
#       --use-rotary-position-embeddings \
#       --no-masked-softmax-fusion \
#       --no-position-embedding \
#       --sequence-parallel \
#       --use-flash-attn \
#       "

NETWORK_SIZE_ARGS=" \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_HEAD} \
       --group-query-attention \
       --num-query-groups ${NUM_QUERY_GROUP} \
       --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
       --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
       --norm-epsilon ${NORM_EPS} \
       --normalization RMSNorm \
       --swiglu \
       --untie-embeddings-and-output-weights \
       --use-rotary-position-embeddings \
       --no-masked-softmax-fusion \
       --no-position-embedding \
       --sequence-parallel \
       --use-flash-attn \
       --transformer-impl local \
       --use-rotary-emb-implement flash_attn \
       --disable-bias-linear \
       --add-qkv-bias \
       "

LOGGING_ARGS=" \
       --log-timers-to-tensorboard \
       --log-validation-ppl-to-tensorboard \
       --log-memory-to-tensorboard \
       "

REGULATIZATION_ARGS=" \
       --attention-dropout ${DROP_OUT} \
       --hidden-dropout ${DROP_OUT} \
       --weight-decay 1e-1 \
       --clip-grad 1.0 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --adam-eps 1e-8 \
       "

TRAINING_ARGS=" \
       --micro-batch-size 1 \
       --global-batch-size 1024 \
       --train-iters 200 \
       --log-interval 1 \
       --optimizer adam \
       "

RECOMPUTE_ARGS="
"


INITIALIZATION_ARGS=" \
       --seed 1024 \
       --init-method-std 0.02 \
       "

LEARNING_RATE_ARGS=" \
       --lr 3e-4 \
       --lr-decay-style cosine \
       --lr-warmup-fraction 0.01 \
       --min-lr 3e-5 \
       "

CHECKPOINTING_ARGS=" \
       --finetune \
       --no-load-optim \
       --no-load-rng \
       --save-interval 1000000 \
       "

MIXED_PRECISION_ARGS=" \
       --bf16 \
       --initial-loss-scale 65536.0 \
       --min-loss-scale 1.0 \
       --loss-scale-window 1000 \
       "

VALIDATION_ARGS=" \
       --eval-interval 10000 \
       --eval-iters 10 \
       "

DATA_ARGS=" \
       --data-path ${DATA_PATH} \
       --split 949,50,1 \
       --seq-length ${MAX_SEQ_LEN} \
       --num-workers 8 \
       --tokenizer-type ChatGLM2Tokenizer \
       --tokenizer-model ${TOKENIZER_PATH} \
       --data-cache-path ${DATA_CACHE_PATH} \
       --dataloader-type single \
       "

CMD="${LAUNCHER} \
       ${SRC_PATH} \
       ${DISTRIBUTED_ARGS} \
       ${NETWORK_SIZE_ARGS} \
       ${LOGGING_ARGS} \
       ${REGULATIZATION_ARGS} \
       ${TRAINING_ARGS} \
       ${RECOMPUTE_ARGS} \
       ${INITIALIZATION_ARGS} \
       ${LEARNING_RATE_ARGS} \
       ${CHECKPOINTING_ARGS} \
       ${MIXED_PRECISION_ARGS} \
       ${VALIDATION_ARGS} \
       ${DATA_ARGS} \
       ${MOE_ARGS} \
       "
echo ${CMD}
${CMD} 2>&1 | tee ${LOG_PATH}
