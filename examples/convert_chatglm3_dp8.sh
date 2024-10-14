HF_MODEL_PATH=$1
MG_MODEL_PATH=$2
PP=1
TP=1

python ../tools/checkpoint/convert.py \
    --model-type GPT \
    --loader chatglm3_hf \
    --saver megatron \
    --target-pipeline-parallel-size ${PP} \
    --target-tensor-parallel-size ${TP} \
    --load-dir ${HF_MODEL_PATH} \
    --save-dir ${MG_MODEL_PATH} \
    --tokenizer-model ${HF_MODEL_PATH} \
    --saver-transformer-impl local \

cp ${HF_MODEL_PATH}/*.json ${MG_MODEL_PATH}/
cp ${HF_MODEL_PATH}/*.py ${MG_MODEL_PATH}/
cp ${HF_MODEL_PATH}/tokenizer.model ${MG_MODEL_PATH}/tokenizer.model