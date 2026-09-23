#!/usr/bin/env bash
set -euo pipefail

CVPR_ROOT=/data_cinema/zqc/CVPR
MODEL_DIR="$CVPR_ROOT/models/Qwen2.5-VL-32B-Instruct-AWQ"
PYTHON=/data_cinema/zqc/VBC/envs/efa/bin/python

# Keep both model files and Hugging Face metadata inside the authorized CVPR
# tree, and avoid a large download if the shared volume becomes tight.
available_kib=$(df -Pk "$CVPR_ROOT" | awk 'NR==2 {print $4}')
if (( available_kib < 60 * 1024 * 1024 )); then
  echo "Insufficient shared-volume headroom for the 20.7 GB model" >&2
  exit 2
fi
mkdir -p "$MODEL_DIR" "$CVPR_ROOT/cache/huggingface"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$CVPR_ROOT/cache/huggingface"
export HF_HUB_CACHE="$CVPR_ROOT/cache/huggingface/hub"
export PYTHONDONTWRITEBYTECODE=1

exec "$PYTHON" -B -c 'from huggingface_hub import snapshot_download; print(snapshot_download(repo_id="Qwen/Qwen2.5-VL-32B-Instruct-AWQ", revision="66c370b", local_dir="/data_cinema/zqc/CVPR/models/Qwen2.5-VL-32B-Instruct-AWQ", max_workers=3), flush=True)'
