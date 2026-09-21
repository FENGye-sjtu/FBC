#!/usr/bin/env bash
set -euo pipefail
source /data_cinema/zqc/CVPR/code/trainingfree/server_env.sh
cd "$CVPR_ROOT/code"
PY=/data_cinema/zqc/VBC/envs/efa/bin/python
# Engineering comparison, NOT original-precision paper replication.
BASE=(--model "$CVPR_ROOT/models/Qwen2.5-VL-7B-Instruct"
 --dataset "$CVPR_ROOT/data/realworldqa/cases.jsonl"
 --upstream "$CVPR_ROOT/upstream/FaithAct"
 --clip "$CVPR_ROOT/models/clip-vit-large-patch14-336"
 --dino "$CVPR_ROOT/models/grounding-dino-base"
 --poll-checkpoint "$CVPR_ROOT/models/faithact_poll/best.pt"
 --quantization nf4 --max-pixels 200704 --max-new-tokens 384 --repairs 3)
"$PY" -B -m trainingfree.run "${BASE[@]}" --methods cot,faithact,online,online_cf \
 --limit 4 --output "$CVPR_ROOT/runs/realworldqa_7b_nf4_pilot4_uncertain_v1"
"$PY" -B -m trainingfree.run "${BASE[@]}" --methods cot,faithact,online,online_cf \
 --limit 16 --output "$CVPR_ROOT/runs/realworldqa_7b_nf4_pilot16_uncertain_v1"
