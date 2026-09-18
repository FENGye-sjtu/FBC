#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=6000 {print $1,$2}' | sort -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then exit 3; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m efa.evaluate_episodes --model "$EFA_ROOT/models/Qwen2-VL-2B-Instruct" --adapter "$EFA_ROOT/runs/gqa_sft_2b_v0" --resampler "$EFA_ROOT/runs/latent_gqa_2b_v0/resampler.pt" --root "$EFA_ROOT/data/derived/gqa_native_review_v1" --output "$EFA_ROOT/runs/gqa_heldout_2b_v0" --policy "$EFA_ROOT/code/configs/policy_engineering.json" --split calibration --limit 1 --engineering-only
