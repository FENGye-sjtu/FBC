#!/usr/bin/env bash
set -euo pipefail

ROOT=/data_cinema/zqc/CVPR/workstreams/faithact75_20260923
PYTHON=/data_cinema/zqc/VBC/envs/efa/bin/python
export PYTHONPATH="$ROOT/snapshot_v27:/data_cinema/zqc/VBC/runtime/qwen3vl_student_v1/python"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/data_cinema/zqc/CVPR/cache/huggingface
export TORCH_HOME=/data_cinema/zqc/CVPR/cache/torch

exec "$PYTHON" -B "$ROOT/code/isolated_bf16_queue.py" \
  --root "$ROOT/queues/qwen3_nf4_control32" \
  --weight-mib 12500 --reserve-mib 1500 --timeout-hours 48 -- \
  "$PYTHON" -B "$ROOT/code/qwen3_baseline.py" \
  --model /data_cinema/zqc/VBC/models/Qwen3-VL-8B-Instruct \
  --dataset /data_cinema/zqc/CVPR/data/vlmevalkit/cases.jsonl \
  --image-root /data_cinema/zqc/CVPR/data/realworldqa \
  --output "$ROOT/runs/qwen3_nf4_layout_v5_control32" \
  --offset 0 --limit 32 --quantization nf4
