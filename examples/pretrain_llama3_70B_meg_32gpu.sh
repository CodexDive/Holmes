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

# Runs the "LLaMA3-8B" parameter model

DATA_DIR="/mnt/nvme/nvme1/sfwang/llama3_data/llama3_wudao"
GPUS_PER_NODE=8
NNODES=$1
NODE_RANK=$2
MASTER_ADDR=$3
MASTER_PORT=12543
MBS=1
ITERS=100
TP=4
PP=8
TDIR="/mnt/nvme/nvme1/sfwang/llama3_data/Meta-Llama-3-8B-Instruct_bav"
MEGAPATH="../"
export PYTHONPATH=$PYTHONPATH:$MEGAPATH
# algorithm args

MODEL_ARGS=" \
    --num-layers 80 \
    --hidden-size 8192 \
    --ffn-hidden-size 28672 \
    --num-attention-heads 64 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1.0e-5 \
    --global-batch-size 512 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --pipline-num-layers-list 9 10 10 10 10 10 11 10 \
    --attention-softmax-in-fp32"

OPT_ARGS=" \
    --lr 1.0e-5 \
    --train-iters $ITERS \
    --lr-decay-iters $ITERS \
    --lr-decay-style cosine \
    --min-lr 1.0e-6 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.008"
    
ALGO_ARGS="$MODEL_ARGS $OPT_ARGS"

# data args

DATA_ARGS=" \
    --data-path $DATA_DIR/wudao_llama3bpe_content_document \
    --tokenizer-type Llama3Tokenizer \
    --tokenizer-model $TDIR \
    --num-workers 8 \
    --split 100,0,0
"

# training args

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

TRAINING_ARGS=" \
    --micro-batch-size $MBS \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --sequence-parallel \
    --bf16
"
RECOMPUTE_ARGS="
       --recompute-granularity full \
       --recompute-method block \
       --recompute-num-layers 9 \
       --recompute-num-layers-list 7 7 6 6 5 5 2 4 \
"
# vendor args

VENDOR_ARGS=" \
    --transformer-impl local \
    --use-distributed-optimizer \
    --use-mcore-models \
    --use-flash-attn
"

OUTPUT_ARGS=" --log-interval 1"

source $ADAPT
run_cmd="torchrun $DISTRIBUTED_ARGS $MEGAPATH/pretrain_gpt.py \
    $ALGO_ARGS \
    $DATA_ARGS \
    $TRAINING_ARGS \
    $RECOMPUTE_ARGS \
    $VENDOR_ARGS \
    $OUTPUT_ARGS"

echo ${run_cmd}
eval ${run_cmd}
