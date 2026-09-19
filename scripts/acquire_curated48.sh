#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
python scripts/acquire_curated_deepeyes.py \
  --selection "$EFA_ROOT/data/selections/deepeyes_next48/selection.json" \
  --source "$EFA_ROOT/data/raw/deepeyesv2_sft" \
  --registry-root "$EFA_ROOT/data/derived" \
  --output "$EFA_ROOT/data/derived/deepeyesv2_curated48_candidates_v1"
