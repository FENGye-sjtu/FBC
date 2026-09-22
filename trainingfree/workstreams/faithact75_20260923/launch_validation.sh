#!/usr/bin/env bash
set -euo pipefail

ROOT=/data_cinema/zqc/CVPR/workstreams/faithact75_20260923
PYTHON=/data_cinema/zqc/VBC/envs/efa/bin/python
export PYTHONPATH="$ROOT/snapshot"

exec "$PYTHON" -B "$ROOT/code/isolated_bf16_queue.py" \
  --root "$ROOT/queues/paper_literal_faithact_validation_bf16" \
  --weight-mib 17000 --reserve-mib 1536 --timeout-hours 48 -- \
  "$PYTHON" -B -m trainingfree.run \
  --model /data_cinema/zqc/CVPR/models/Qwen2.5-VL-7B-Instruct \
  --dataset /data_cinema/zqc/CVPR/data/vlmevalkit/cases.jsonl \
  --image-root /data_cinema/zqc/CVPR/data/realworldqa \
  --output "$ROOT/runs/paper_literal_faithact_validation128_bf16" \
  --methods cot,faithact --offset 128 --limit 128 \
  --device auto --quantization none --max-new-tokens 512 \
  --max-pixels 1003520 --cot-style faithact_paper_cot_literal \
  --upstream /data_cinema/zqc/CVPR/upstream/FaithAct \
  --clip /data_cinema/zqc/CVPR/models/clip-vit-large-patch14-336 \
  --dino /data_cinema/zqc/CVPR/models/grounding-dino-base \
  --poll-checkpoint /data_cinema/zqc/CVPR/models/faithact_poll/best.pt \
  --tool-device cpu --faithact-box-threshold 0.25 \
  --faithact-text-threshold 0.20 --faithact-prompt released_code
