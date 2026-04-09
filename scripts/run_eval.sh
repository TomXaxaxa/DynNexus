#!/bin/bash

main_dir=$(cd "$(dirname "$0")/../.." && pwd)
echo "main_dir: $main_dir"
checkpoint_dir="./checkpoints/Nexus/checkpoint-final"

ulimit -n 99999

test_data_dirs=(
    ./data/test
)

test_data_dirs_json="[" 
num_dirs=${#test_data_dirs[@]} 
i=0
for dir in "${test_data_dirs[@]}"; do
    test_data_dirs_json+="\"$dir\"" 
    i=$((i + 1))
    if [ "$i" -lt "$num_dirs" ]; then
        test_data_dirs_json+="," 
    fi
done
test_data_dirs_json+="]" 

echo "test_data_dirs: $test_data_dirs_json"

python scripts/evaluate.py \
    eval.checkpoint_path=$checkpoint_dir \
    eval.data_paths_lst=$test_data_dirs_json \
    eval.num_subdirs=null \
    eval.num_test_instances=6 \
    eval.window_style=sampled \
    eval.batch_size=128 \
    eval.prediction_length=512 \
    eval.results_dir=./eval_results/Nexus-test/ \
    eval.overwrite=true \
    eval.device=cuda:6 \
    eval.save_eval_artifacts=false \