#!/usr/bin/env bash
set -euo pipefail

ROOT=/data_cinema/zqc/CVPR/workstreams/faithact75_20260923
PYTHON=/data_cinema/zqc/VBC/envs/efa/bin/python
export PYTHONPATH="$ROOT/snapshot_v32controller"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/data_cinema/zqc/CVPR/cache/huggingface
export TORCH_HOME=/data_cinema/zqc/CVPR/cache/torch

exec "$PYTHON" -B "$ROOT/code/isolated_bf16_queue.py" \
  --root "$ROOT/queues/v32_nf4_pilot8" \
  --weight-mib 7000 --reserve-mib 1500 --min-each-free-mib 8500 \
  --timeout-hours 48 -- \
  "$PYTHON" -B -m trainingfree.workstreams.faithact75_20260923.run_v32 \
  --model /data_cinema/zqc/CVPR/models/Qwen2.5-VL-7B-Instruct \
  --dataset /data_cinema/zqc/CVPR/data/vlmevalkit/cases.jsonl \
  --image-root /data_cinema/zqc/CVPR/data/realworldqa \
  --output "$ROOT/runs/v32_answer_only_stepwise_nf4_pilot8" \
  --methods evidence_first --limit 8 --device cuda:0 --quantization nf4 \
  --max-new-tokens 512 --max-pixels 1003520 --max-steps 6 --repairs 3 \
  --repair-style atomic_v1
