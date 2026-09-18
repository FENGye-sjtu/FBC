#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
repo='Qwen/Qwen2-VL-2B-Instruct'
root=Path('/data_cinema/zqc/VBC/models/Qwen2-VL-2B-Instruct')
info=HfApi(endpoint='https://hf-mirror.com').model_info(repo)
snapshot_download(repo,revision=info.sha,local_dir=root,
    allow_patterns=['*.json','*.safetensors','*.txt','*.model'],max_workers=2)
(root/'source_revision.json').write_text(json.dumps(dict(repository=repo,revision=info.sha)))
print('MODEL_2B_READY',info.sha,flush=True)
PY
