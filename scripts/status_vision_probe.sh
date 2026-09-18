#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
ps -eo pid,etime,args | awk '/VBC.*vision_precision_2b|python -m efa.vision_probe/ && !/awk/'
python - <<'PY'
import json
from pathlib import Path
root=Path('/data_cinema/zqc/VBC')
for label in ['vision_precision_2b_v0','vision_precision_2b_v1','vision_precision_2b_v2']:
    out=root/'runs'/label
    p=out/'results.jsonl'
    if p.exists():
        for line in p.read_text().splitlines():
            r=json.loads(line);r.pop('calls',None);print(label,json.dumps(r))
    print(label,'COMPLETE' if (out/'summary.json').exists() else 'PENDING')
    log=root/'logs'/(label+'.log')
    if log.exists():print('\n'.join(line[:800] for line in log.read_text(errors='replace').splitlines()[-3:]))
PY
