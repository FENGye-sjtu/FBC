#!/usr/bin/env bash
set -euo pipefail

CVPR_ROOT=/data_cinema/zqc/CVPR
ROOT="$CVPR_ROOT/workstreams/faithact75_20260923"
PYTHON=/data_cinema/zqc/VBC/envs/efa/bin/python
export PYTHONPATH="$CVPR_ROOT/vendor/autoawq:$ROOT/snapshot_v33awq"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$CVPR_ROOT/cache/huggingface"
export TORCH_HOME="$CVPR_ROOT/cache/torch"

exec "$PYTHON" -B "$ROOT/code/isolated_bf16_queue.py" \
  --root "$ROOT/queues/qwen25_32b_awq_smoke8" \
  --weight-mib 21000 --reserve-mib 1024 --timeout-hours 48 -- \
  "$PYTHON" -B -m trainingfree.run \
  --model "$CVPR_ROOT/models/Qwen2.5-VL-32B-Instruct-AWQ" \
  --dataset "$CVPR_ROOT/data/vlmevalkit/cases.jsonl" \
  --image-root "$CVPR_ROOT/data/realworldqa" \
  --output "$ROOT/runs/qwen25_32b_awq_layout_v5_smoke8" \
  --methods cot --cot-style evidence_layout_v5 \
  --limit 8 --device auto --quantization awq \
  --max-new-tokens 512 --max-pixels 1003520
