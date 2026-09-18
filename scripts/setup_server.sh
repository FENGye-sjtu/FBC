#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$WANDB_DIR"
if ! python -m pip --version >/dev/null 2>&1; then
  curl -fL --retry 3 --connect-timeout 10 --max-time 120 https://bootstrap.pypa.io/get-pip.py -o "$EFA_ROOT/tools/get-pip.py"
  python "$EFA_ROOT/tools/get-pip.py" --index-url https://pypi.tuna.tsinghua.edu.cn/simple
fi
python -m pip install -e "$EFA_ROOT/code[data]" --index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m unittest discover -s "$EFA_ROOT/code/tests" -v
printf 'DATA_ENV_READY\n'
