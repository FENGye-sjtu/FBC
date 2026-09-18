#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=6000 {print $1,$2}' | sort -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then exit 3; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m efa.diagnose_reader --model "$EFA_ROOT/models/Qwen2-VL-2B-Instruct" --adapter "$EFA_ROOT/runs/episode_sft_2b_v0" --region-adapter "$EFA_ROOT/runs/region_sft_2b_v1" --root "$EFA_ROOT/data/derived/zwz_pilot_v1" --evaluation "$EFA_ROOT/runs/heldout_2b_v0/results.jsonl" --output "$EFA_ROOT/runs/reader_diagnostic_2b_v0"
