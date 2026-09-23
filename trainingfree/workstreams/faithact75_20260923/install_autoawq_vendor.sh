#!/usr/bin/env bash
set -euo pipefail

CVPR_ROOT=/data_cinema/zqc/CVPR
PYTHON=/data_cinema/zqc/VBC/envs/efa/bin/python
mkdir -p "$CVPR_ROOT/vendor/autoawq" "$CVPR_ROOT/cache/pip" "$CVPR_ROOT/tmp"
export PIP_CACHE_DIR="$CVPR_ROOT/cache/pip"
export TMPDIR="$CVPR_ROOT/tmp"
export PYTHONDONTWRITEBYTECODE=1

# Existing torch/transformers/triton/accelerate remain untouched in the
# read-only VBC environment. Only the missing AWQ package and zstandard go to
# the CVPR vendor tree; runtime PYTHONPATH opts into them per experiment.
exec "$PYTHON" -B -m pip install --no-deps --target "$CVPR_ROOT/vendor/autoawq" \
  autoawq==0.2.9 zstandard
