#!/usr/bin/env bash
set -euo pipefail

ROOT=/data_cinema/zqc/CVPR/workstreams/faithact75_20260923
PYTHON=/data_cinema/zqc/VBC/envs/efa/bin/python
export PYTHONPATH="$ROOT/snapshot_v24b"

exec "$PYTHON" -B "$ROOT/code/isolated_bf16_queue.py" \
  --root "$ROOT/queues/v24b_blind_repair_3b_nf4_smoke" \
  --weight-mib 4500 --reserve-mib 3000 --timeout-hours 24 -- \
  "$PYTHON" -B -m trainingfree.workstreams.faithact75_20260923.run_v24 \
  --model /data_cinema/zqc/VBC/models/Qwen2.5-VL-3B-Instruct \
  --dataset /data_cinema/zqc/CVPR/data/vlmevalkit/cases.jsonl \
  --image-root /data_cinema/zqc/CVPR/data/realworldqa \
  --output "$ROOT/runs/v24b_blind_repair_3b_nf4_smoke2" \
  --methods evidence_first --limit 2 --device cuda:0 --quantization nf4 \
  --max-new-tokens 384 --max-pixels 200704 --max-steps 6 --repairs 3 \
  --repair-style atomic_v1

