#!/usr/bin/env bash
set -euo pipefail

ROOT=/data_cinema/zqc/CVPR/workstreams/faithact75_20260923
PYTHON=/data_cinema/zqc/VBC/envs/efa/bin/python
export PYTHONPATH="$ROOT/snapshot_v30prompt"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/data_cinema/zqc/CVPR/cache/huggingface
export TORCH_HOME=/data_cinema/zqc/CVPR/cache/torch

exec "$PYTHON" -B "$ROOT/code/isolated_bf16_queue.py" \
  --root "$ROOT/queues/prompt_pair_v6_nf4_dev128" \
  --weight-mib 7000 --reserve-mib 1500 --min-each-free-mib 8500 \
  --timeout-hours 48 -- \
  "$PYTHON" -B -m trainingfree.workstreams.faithact75_20260923.run_prompt_pair_v6
