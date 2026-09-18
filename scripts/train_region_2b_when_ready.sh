#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
for attempt in $(seq 1 60); do
  if [ -f "$EFA_ROOT/models/Qwen2-VL-2B-Instruct/source_revision.json" ]; then
    exec bash "$EFA_ROOT/code/scripts/train_region_2b.sh"
  fi
  printf 'Waiting for the 2B model download, check %s/60\n' "$attempt"
  sleep 30
done
exit 3
