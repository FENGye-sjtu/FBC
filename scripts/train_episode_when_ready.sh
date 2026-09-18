#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
for attempt in $(seq 1 60); do
  if [ -f "$EFA_ROOT/runs/latent_reader_2b_v1/metrics.json" ]; then
    exec bash "$EFA_ROOT/code/scripts/train_episode_2b.sh"
  fi
  printf 'Waiting for native reader alignment, check %s/60\n' "$attempt"
  sleep 30
done
exit 3
