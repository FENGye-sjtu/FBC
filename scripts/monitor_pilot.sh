#!/usr/bin/env bash
set -eu
for attempt in $(seq 1 6); do
  bash /data_cinema/zqc/VBC/code/scripts/status_pilot.sh
  sleep 30
done
