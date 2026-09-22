#!/usr/bin/env bash
set -euo pipefail

ROOT=/data_cinema/zqc/CVPR
SNAPSHOT="$ROOT/code_snapshots/v21_repair_guard"
source "$SNAPSHOT/trainingfree/server_env.sh"
export PYTHONPATH="$SNAPSHOT"

# The BF16 worker owns GPUs 0/1 and its tool GPU 3. Use only GPU 2 while it fits.
exec 9>"$ROOT/runtime/gpu-2.lock"
flock -n 9 || { echo 'GPU 2 is reserved by another CVPR job'; exit 2; }
free_mib=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 7500 )); then
  echo "Insufficient free GPU 2 memory: ${free_mib} MiB"
  exit 2
fi
export CUDA_VISIBLE_DEVICES=2
export CVPR_GPU_BUDGET_MIB=$((free_mib - 512))

exec /data_cinema/zqc/VBC/envs/efa/bin/python -B -m trainingfree.worker \
  --directory "$ROOT/runtime/queues/v21_repair_guard" --idle-seconds 10
