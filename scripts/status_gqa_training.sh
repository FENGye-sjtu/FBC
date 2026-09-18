#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader
ps -eo pid,etime,args | awk '/VBC.*(gqa_sft_2b|latent_gqa_2b|gqa_training_2b|gqa_mixed_training_2b)/ && !/awk/'
python - <<'PY'
import json
from pathlib import Path
root=Path('/data_cinema/zqc/VBC')
for label in ['gqa_training_2b_v0','gqa_mixed_training_2b_v1']:
    log=root/'logs'/(label+'.log')
    if log.exists():print(label,'\n'.join(line[:700] for line in log.read_text(errors='replace').splitlines()[-4:]))
for name in ['data/derived/gqa_sft_2b_v0/export_manifest.json','runs/gqa_sft_2b_v0/all_results.json','runs/latent_gqa_2b_v0/metrics.json',
             'data/derived/gqa_sft_2b_v1/export_manifest.json','runs/gqa_sft_2b_v1/precision_policy.json',
             'runs/gqa_sft_2b_v1/all_results.json','runs/latent_gqa_2b_v1/metrics.json','runs/gqa_heldout_2b_v1/summary.json']:
    p=root/name
    if p.exists():
        value=json.loads(p.read_text());value.pop('losses',None);print(name,json.dumps(value))
    else:print(name,'PENDING')
PY
