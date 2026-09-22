#!/usr/bin/env bash
set -euo pipefail

ROOT=/data_cinema/zqc/CVPR/workstreams/faithact75_20260923
PYTHON=/data_cinema/zqc/VBC/envs/efa/bin/python
export PYTHONPATH="$ROOT/snapshot_v25"

exec "$PYTHON" -B "$ROOT/code/isolated_bf16_queue.py" \
  --root "$ROOT/queues/v25_blind_repair_nf4" \
  --weight-mib 7500 --reserve-mib 2048 --timeout-hours 48 -- \
  "$PYTHON" -B -m trainingfree.workstreams.faithact75_20260923.run_v25 \
  --model /data_cinema/zqc/CVPR/models/Qwen2.5-VL-7B-Instruct \
  --dataset /data_cinema/zqc/CVPR/data/vlmevalkit/cases.jsonl \
  --image-root /data_cinema/zqc/CVPR/data/realworldqa \
  --output "$ROOT/runs/v25_blind_repair_nf4_pilot4" \
  --methods evidence_first --limit 4 --device cuda:0 --quantization nf4 \
  --max-new-tokens 512 --max-pixels 1003520 --max-steps 6 --repairs 3 \
  --repair-style atomic_v1
