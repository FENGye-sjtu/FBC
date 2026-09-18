#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
python -m pip install 'torch==2.6.0' 'torchvision==0.21.0' --index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -e '.[model]' 'bitsandbytes>=0.45' --index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", local_dir="/data_cinema/zqc/VBC/models/Qwen2.5-VL-3B-Instruct", allow_patterns=["*.json","*.safetensors","*.txt","*.model"], max_workers=2)'
printf 'MODEL_READY\n'
