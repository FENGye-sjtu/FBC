#!/usr/bin/env bash
set -eu
cd /data_cinema/zqc/VBC
df -h . /data_wyn/zqc/VBC
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader
for name in setup model data sft_setup reader_sft reader_sft_v1 model_smoke; do
  if [ -f "logs/$name.log" ]; then
    printf '\nJob %s\n' "$name"
    tail -n 8 "logs/$name.log"
  fi
done
if [ -x envs/efa/bin/python ]; then
  envs/efa/bin/python -m pip --version || true
fi
