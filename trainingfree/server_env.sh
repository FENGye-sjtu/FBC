#!/usr/bin/env bash
# Source this file BEFORE running Python or downloading anything.
export CVPR_ROOT=/data_cinema/zqc/CVPR
export HF_HOME="$CVPR_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="$CVPR_ROOT/cache/xdg"
export TORCH_HOME="$CVPR_ROOT/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$CVPR_ROOT/cache/inductor"
export TRITON_CACHE_DIR="$CVPR_ROOT/cache/triton"
export CUDA_CACHE_PATH="$CVPR_ROOT/cache/cuda"
export MPLCONFIGDIR="$CVPR_ROOT/cache/matplotlib"
export PIP_CACHE_DIR="$CVPR_ROOT/cache/pip"
export TMPDIR="$CVPR_ROOT/tmp"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH="$CVPR_ROOT/code"
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" \
 "$XDG_CACHE_HOME" "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
 "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$PIP_CACHE_DIR" "$CVPR_ROOT/logs" "$CVPR_ROOT/runs"
