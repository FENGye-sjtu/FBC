#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
python -m efa.deepeyes --source "$EFA_ROOT/data/raw/deepeyesv2_sft" --output "$EFA_ROOT/data/derived/deepeyesv2_mixed_v1" --limit 128 --seed 20260919
