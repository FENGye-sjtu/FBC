#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('honglyhly/DeepEyesV2_SFT',repo_type='dataset',
    revision='cee88d9e9c494729f719ccca5de5aeb940501a35',
    local_dir='/data_cinema/zqc/VBC/data/raw/deepeyesv2_sft',
    allow_patterns=['json/sft_part_0.json','images.zip'],max_workers=2)
print('DEEPEYESV2_SOURCE_READY',flush=True)
PY
