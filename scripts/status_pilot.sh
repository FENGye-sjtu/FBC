#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader
ps -eo pid,etime,args | awk '/VBC.*region_sft_pilot_v2/ && !/awk/'
tail -n 8 "$EFA_ROOT/logs/region_sft_pilot_v2_lowmem.log"
python - <<'PY'
import json
from pathlib import Path
root=Path('/data_cinema/zqc/VBC')
images=root/'data/derived/region_sft_pilot_v2/images'
print('exported_crop_files',sum(1 for _ in images.glob('*.png')) if images.exists() else 0)
for name in ['data/derived/region_sft_pilot_v2/export_manifest.json',
             'data/derived/deepeyesv2_mixed_v1/manifest.json',
             'runs/region_sft_pilot_v2/all_results.json',
             'runs/latent_reader_pilot_v2/metrics.json']:
    path=root/name
    if path.exists():
        value=json.loads(path.read_text());value.pop('losses',None)
        print(name,json.dumps(value))
    else:print(name,'PENDING')
PY
