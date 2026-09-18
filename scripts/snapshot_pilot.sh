#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
python -m efa.snapshot --source "$EFA_ROOT/data/derived/zwz_pilot_v1" --output "$EFA_ROOT/data/derived/zwz_pilot_128_v1" --limit 128
