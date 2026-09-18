#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=6500 {print $1,$2}' | sort -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then
  printf 'WAITING_FOR_GPU\n'
  exit 3
fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m efa.train_latent --model "$EFA_ROOT/models/Qwen2.5-VL-3B-Instruct" --adapter "$EFA_ROOT/runs/reader_sft_smoke_v1" --root "$EFA_ROOT/data/derived/zwz_seed_v0" --output "$EFA_ROOT/runs/latent_reader_smoke_v0" --steps 40 --engineering-only
