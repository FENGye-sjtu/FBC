#!/usr/bin/env bash
set -euo pipefail

ROOT=/data_cinema/zqc/CVPR/workstreams/faithact75_20260923
PYTHON=/data_cinema/zqc/VBC/envs/efa/bin/python
export PYTHONPATH="$ROOT/snapshot_v29:/data_cinema/zqc/VBC/runtime/qwen3vl_student_v1/python"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/data_cinema/zqc/CVPR/cache/huggingface
export TORCH_HOME=/data_cinema/zqc/CVPR/cache/torch

exec "$PYTHON" -B "$ROOT/code/isolated_bf16_queue.py" \
  --root "$ROOT/queues/blind_view_probe" \
  --weight-mib 7000 --reserve-mib 1500 --min-each-free-mib 10000 \
  --timeout-hours 12 -- \
  "$PYTHON" -B -m trainingfree.workstreams.faithact75_20260923.probe_blind_views \
  --model /data_cinema/zqc/VBC/models/Qwen3-VL-8B-Instruct \
  --dataset /data_cinema/zqc/CVPR/data/vlmevalkit/cases.jsonl \
  --image-root /data_cinema/zqc/CVPR/data/realworldqa \
  --trace "$ROOT/runs/v29_dual_blind_nf4_pilot16_slow/evidence_first.jsonl" \
  --output "$ROOT/analysis/blind_view_probe_case0.json" --id 0
