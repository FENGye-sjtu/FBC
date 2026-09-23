#!/usr/bin/env bash
set -euo pipefail

ROOT=/data_cinema/zqc/CVPR/workstreams/faithact75_20260923
PYTHON=/data_cinema/zqc/VBC/envs/efa/bin/python
export PYTHONPATH="$ROOT/snapshot_v31prompt"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/data_cinema/zqc/CVPR/cache/huggingface
export TORCH_HOME=/data_cinema/zqc/CVPR/cache/torch

exec "$PYTHON" -B "$ROOT/code/isolated_bf16_queue.py" \
  --root "$ROOT/queues/option_scores_nf4_dev128" \
  --weight-mib 7000 --reserve-mib 1500 --min-each-free-mib 8500 \
  --timeout-hours 48 -- \
  "$PYTHON" -B -m trainingfree.workstreams.faithact75_20260923.run_option_scores \
  --model /data_cinema/zqc/CVPR/models/Qwen2.5-VL-7B-Instruct \
  --dataset /data_cinema/zqc/CVPR/data/vlmevalkit/cases.jsonl \
  --image-root /data_cinema/zqc/CVPR/data/realworldqa \
  --output "$ROOT/runs/qwen25_nf4_option_scores_dev128.jsonl" \
  --offset 0 --limit 128 --device cuda:0 --quantization nf4 \
  --max-pixels 1003520
