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
export MACA_SMALL_PAGESIZE_ENABLE=1
export MHA_USE_BLAS=ON
export MHA_BWD_NO_ATOMIC_F64=1
export SET_DEVICE_NUMA_PREFERRED=1
export MAX_JOBS=20


# Distributed training variables
NNODES=4
GPUS_PER_NODE=8
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
NODE_RANK=$1
MASTER_PORT=12453
MASTER_ADDR="10.20.34.9"

# Parallelism variables
TP=4
PP=8
DP=$((${GPU_NUM}/${TP}/${PP}))

# Network size variables
MODEL_SIZE=70

if   [ ${MODEL_SIZE} == 7 ];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=2; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == 13 ];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == 70 ];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == 130 ];  then HIDDEN_SIZE=12288;  NUM_HEAD=96; NUM_QUERY_GROUP=8;  NUM_LAYERS=10; FFN_HIDDEN_SIZE=31232; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == "tiny" ]; then HIDDEN_SIZE=128;  NUM_HEAD=4; NUM_QUERY_GROUP=4; NUM_LAYERS=4; FFN_HIDDEN_SIZE=512; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

DROP_OUT=0.0
MAX_SEQ_LEN=4096
MAX_POSITION_EMBEDDINGS=4096

# Paths
BASE_PATH=/mnt/nvme/nvme1/sfwang/llama2_data
SRC_PATH=../pretrain_llama.py

LOG_NAME=llama2-70b_pretrain_WS${WORLD_SIZE}_TP${TP}_PP${PP}
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p ${BASE_PATH}/log/${LOG_NAME}

#DATA_PATH=${BASE_PATH}/data/pile_wikipedia_demo
DATA_PATH=${BASE_PATH}/oscar-en-10k-meg-llama_text_document
#DATA_PATH=/software/home/sfwang/code/FlagScale/data/pile_wikipedia_demo
DATA_CACHE_PATH="./data_cache/${LOG_NAME}"
mkdir -p ${DATA_CACHE_PATH}

SAVE_PATH=${BASE_PATH}/checkpoint/${LOG_NAME}
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p ${BASE_PATH}/log/${LOG_NAME}

#TOKENIZER_PATH=${BASE_PATH}/data/tokenizer.model
TOKENIZER_PATH=${BASE_PATH}/tokenizer.model

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
       --use-distributed-optimizer \
       --sequence-parallel \
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
       --use-rotary-position-embeddings \
       --no-position-embedding \
       --swiglu \
       --untie-embeddings-and-output-weights \
       --transformer-impl local \
       --sequence-parallel \
       --use-flash-attn \
       --pipline-num-layers-list 9 9 10 10 10 11 11 10 \
       --recompute-granularity full \
       --recompute-method block \
       --recompute-num-layers 4 \
       --recompute-num-layers-list 2 0 1 0 0 0 0 0 \
       "

LOGGING_ARGS=""

REGULATIZATION_ARGS=" \
       --attention-dropout ${DROP_OUT} \
       --hidden-dropout ${DROP_OUT} \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --adam-eps 1e-8 \
       "

TRAINING_ARGS=" \
       --micro-batch-size 1 \
       --global-batch-size 1024 \
       --train-iters 200 \
       --disable-bias-linear \
       --log-interval 1 \
       "

INITIALIZATION_ARGS=" \
       --seed 1024 \
       --init-method-std 0.02 \
       "

LEARNING_RATE_ARGS=" \
       --lr 1e-4 \
       --lr-decay-style cosine \
       --lr-warmup-fraction 0.01 \
       --min-lr 1e-5 \
       "

CHECKPOINTING_ARGS=""

MIXED_PRECISION_ARGS=" \
       --bf16 \
       "

VALIDATION_ARGS=" \
       --eval-interval 10000 \
       --eval-iters 10 \
       "

DATA_ARGS=" \
       --data-path ${DATA_PATH} \
       --split 1 \
       --seq-length ${MAX_SEQ_LEN} \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model ${TOKENIZER_PATH} \
       "

CMD="${LAUNCHER} \
       ${SRC_PATH} \
       ${DISTRIBUTED_ARGS} \
       ${NETWORK_SIZE_ARGS} \
       ${LOGGING_ARGS} \
       ${REGULATIZATION_ARGS} \
       ${TRAINING_ARGS} \
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
