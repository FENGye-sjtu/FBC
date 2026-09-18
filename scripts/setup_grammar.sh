#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
python -m pip install lm-format-enforcer==0.10.11 --index-url https://pypi.tuna.tsinghua.edu.cn/simple
