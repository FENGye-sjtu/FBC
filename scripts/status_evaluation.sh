#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader
ps -eo pid,etime,args | awk '/VBC.*heldout_2b|python -m efa.evaluate/ && !/awk/'
python - <<'PY'
import json
from pathlib import Path
root=Path('/data_cinema/zqc/VBC')
log=root/'logs/heldout_2b_v0.log'
if log.exists():
    print('\n'.join(log.read_text(errors='replace').splitlines()[-12:]))
out=root/'runs/heldout_2b_v0'
for name in ['summary.json','results.jsonl']:
    path=out/name
    if not path.exists():
        print(name,'PENDING')
    elif name=='summary.json':
        print(path.read_text())
    else:
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        print('completed_examples',len(rows))
        for row in rows:
            print(json.dumps(dict(id=row['id'],reference=row['reference'],
                readers={k:{f:v[f] for f in ['answer','correct','reference_nll_per_token']} for k,v in row['readers'].items()},
                efa=row.get('efa'))))
PY
