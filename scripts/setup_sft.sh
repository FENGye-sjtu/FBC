#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
python -m pip install 'llamafactory==0.9.3' 'transformers==4.51.3' 'peft==0.15.2' 'accelerate<1.8' --index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -c 'from transformers import Qwen2_5_VLForConditionalGeneration; import peft; import llamafactory; print("SFT_IMPORT_OK", peft.__version__)'
printf 'SFT_ENV_READY\n'
