#!/bin/bash
# Ref hfd from: https://hf-mirror.com/

sudo apt install aria2c
export HF_ENDPOINT=https://hf-mirror.com

# Set the parameters
repo_id="HiDream-ai/ReCo-Bench"
hf_username="YOUR HF NAME"
hf_token="YOUR HF TOKEN"
tool="aria2c"
threads="10"
concurrent_file='1'
dir='YOUR_ReCo-Bench_CACHE_PATH_DIR'
# exclude_pattern="*.safetensors"


bash hfd.sh $model_id \
    --hf_username $hf_username \
    --hf_token $hf_token \
    --tool $tool \
    -x $threads \
    -j $concurrent_file \
    --local-dir $dir \
    --dataset
    # --exclude $exclude_pattern \

