#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
python -m efa.zwz --metadata "$EFA_ROOT/data/raw_metadata/zwz/train.parquet" --output "$EFA_ROOT/data/derived/zwz_pilot_v1" --limit 1024
