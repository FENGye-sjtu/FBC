#!/usr/bin/env bash
set -euo pipefail

ROOT=/data_cinema/zqc/CVPR/workstreams/faithact75_20260923
PYTHON=/data_cinema/zqc/VBC/envs/efa/bin/python
export PYTHONPATH="$ROOT/snapshot_v26"

exec "$PYTHON" -B "$ROOT/code/isolated_bf16_queue.py" \
  --root "$ROOT/queues/v26_layout_anchor_nf4" \
  --weight-mib 7000 --reserve-mib 1500 --timeout-hours 48 -- \
  "$PYTHON" -B -m trainingfree.workstreams.faithact75_20260923.run_v26 \
  --model /data_cinema/zqc/CVPR/models/Qwen2.5-VL-7B-Instruct \
  --dataset /data_cinema/zqc/CVPR/data/vlmevalkit/cases.jsonl \
  --image-root /data_cinema/zqc/CVPR/data/realworldqa \
  --output "$ROOT/runs/v26_layout_anchor_nf4_pilot16" \
  --methods cot,evidence_first --limit 16 --device cuda:0 --quantization nf4 \
  --cot-style evidence_layout_v5 \
  --max-new-tokens 512 --max-pixels 1003520 --max-steps 6 --repairs 3 \
  --repair-style atomic_v1
