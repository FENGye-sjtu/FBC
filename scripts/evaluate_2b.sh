#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=6000 {print $1,$2}' | sort -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then exit 3; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m efa.evaluate --model "$EFA_ROOT/models/Qwen2-VL-2B-Instruct" --adapter "$EFA_ROOT/runs/episode_sft_2b_v0" --resampler "$EFA_ROOT/runs/latent_episode_2b_v0/resampler.pt" --root "$EFA_ROOT/data/derived/zwz_pilot_v1" --output "$EFA_ROOT/runs/heldout_2b_v0" --policy "$EFA_ROOT/code/configs/policy_engineering.json" --limit 16 --episodes 2 --engineering-only
