#!/bin/bash

DEBUG=0
while getopts "d" flag; do
        case "${flag}" in
                d) DEBUG=1;;
        esac
done
shift $((OPTIND - 1))

train_data_dirs=(
      ./data/train
)

train_data_dirs_json="[" 
num_dirs=${#train_data_dirs[@]} 
i=0
for dir in "${train_data_dirs[@]}"; do
    train_data_dirs_json+="\"$dir\"" 
    i=$((i + 1))
    if [ "$i" -lt "$num_dirs" ]; then
        train_data_dirs_json+="," 
    fi
done
train_data_dirs_json+="]" 

echo "train_data_dirs: $train_data_dirs_json"

ulimit -n 999999
if [ "$DEBUG" -eq 0 ]; then

        TOTAL_CORES=$(nproc)
        CORES_PER_GROUP=$(( $TOTAL_CORES / 2 ))
        CORES_PER_JOB=$(( $CORES_PER_GROUP / 4 ))

        CUDA_DEVICES=7
        NUM_DEVICES=$(echo "$CUDA_DEVICES" | tr -d ' ' | tr ',' '\n' | wc -l)

        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES OMP_NUM_THREADS=$CORES_PER_JOB python3 -m torch.distributed.run \
                --nproc-per-node $NUM_DEVICES \
                --master-port 29536 \
                scripts/train.py \
                run_name=NexusOrigin \
                train_data_dirs=$train_data_dirs_json \
                shuffle_buffer_length=100 \
                train.per_device_train_batch_size=1024 \
                train.max_steps=100000 \
                train.save_steps=10000 \
                train.log_steps=100 \
                train.torch_compile=true \
                train.output_dir=./checkpoints/Nexus-Original \
                "$@"
else  # this mode allows for breakpoints inside model code
        CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
                run_name=DEBUG \
                train_data_dirs=$train_data_dirs_json \
                shuffle_buffer_length=100 \
                train.per_device_train_batch_size=16 \
                train.ddp_backend=null \
                train.torch_compile=false \
                train.output_dir=./checkpoints/
                "$@"
fi