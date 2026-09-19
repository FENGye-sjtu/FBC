#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
python - <<'PY'
import json
from pathlib import Path
import subprocess
root=Path('/data_cinema/zqc/VBC')
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader'],text=True))
for line in subprocess.check_output(['ps','-eo','pid,etime,args'],text=True).splitlines():
    if 'python -m efa.' in line and any(s in line for s in ['reader_objectives_2b','gqa_reader_objectives_eval']):print(line)
log=root/'logs/reader_objectives_2b_v1.log'
if log.exists():print('\n'.join(line[:600] for line in log.read_text(errors='replace').splitlines()[-3:]))
for name in ['runs/reader_objectives_2b_v1/metrics.json','runs/gqa_reader_objectives_eval_v1/summary.json']:
    p=root/name
    if not p.exists():print(name,'PENDING');continue
    record=json.loads(p.read_text())
    print(name,json.dumps({k:v for k,v in record.items() if k not in ['encoder_config','checkpoint','config']}))
results=root/'runs/gqa_reader_objectives_eval_v1/results.jsonl'
if results.exists():
    for line in results.read_text().splitlines():
        r=json.loads(line)
        print(json.dumps(dict(id=r['id'],reference=r['reference'],
            compact=[a['compact'] for a in r['reads']],efa=r['efa'])))
PY
