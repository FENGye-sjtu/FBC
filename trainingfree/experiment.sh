#!/usr/bin/env bash
set -euo pipefail
source /data_cinema/zqc/CVPR/code/trainingfree/server_env.sh
cd "$CVPR_ROOT/code"
PY=/data_cinema/zqc/VBC/envs/efa/bin/python
BASE=(--model "$CVPR_ROOT/models/Qwen2.5-VL-7B-Instruct"
 --dataset "$CVPR_ROOT/data/realworldqa/cases.jsonl"
 --upstream "$CVPR_ROOT/upstream/FaithAct"
 --clip "$CVPR_ROOT/models/clip-vit-large-patch14-336"
 --dino "$CVPR_ROOT/models/grounding-dino-base"
 --poll-checkpoint "$CVPR_ROOT/models/faithact_poll/best.pt"
 --methods cot,faithact,online,online_cf --max-new-tokens 768)
# A small paired run establishes functionality and initial accuracy, not paper parity.
"$PY" -B -m trainingfree.run "${BASE[@]}" --limit 16 --output "$CVPR_ROOT/runs/realworldqa_7b_pilot16_v1"
# Baseline reproduction proceeds separately; method tuning does not block it.
"$PY" -B -m trainingfree.run "${BASE[@]}" --methods cot,faithact --output "$CVPR_ROOT/runs/realworldqa_7b_reproduction_v1"
