#!/usr/bin/env bash
set -euo pipefail

ROOT=/data_cinema/zqc/CVPR/workstreams/faithact75_20260923
PYTHON=/data_cinema/zqc/VBC/envs/efa/bin/python
export PYTHONPATH="$ROOT/snapshot_v30fixed"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/data_cinema/zqc/CVPR/cache/huggingface
export TORCH_HOME=/data_cinema/zqc/CVPR/cache/torch

exec "$PYTHON" -B "$ROOT/code/isolated_bf16_queue.py" \
  --root "$ROOT/queues/v30_formatted_guard_smoke" \
  --weight-mib 7000 --reserve-mib 1500 --min-each-free-mib 8500 \
  --timeout-hours 48 -- \
  "$PYTHON" -B -m trainingfree.workstreams.faithact75_20260923.run_v28 \
  --model /data_cinema/zqc/CVPR/models/Qwen2.5-VL-7B-Instruct \
  --dataset /data_cinema/zqc/CVPR/data/vlmevalkit/cases.jsonl \
  --image-root /data_cinema/zqc/CVPR/data/realworldqa \
  --output "$ROOT/runs/v30_formatted_guard_smoke_nf4" \
  --methods evidence_first --limit 1 --device cuda:0 --quantization nf4 \
  --max-new-tokens 512 --max-pixels 1003520 --max-steps 6 --repairs 3 \
  --repair-style atomic_v1
