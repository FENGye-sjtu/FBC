#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=7500 {print $1,$2}' | sort -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then exit 3; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m efa.vision_probe --model "$EFA_ROOT/models/Qwen2-VL-2B-Instruct" --root "$EFA_ROOT/data/derived/gqa_native_review_v1" --id gqa-native-6019 --output "$EFA_ROOT/runs/vision_precision_2b_v0"
