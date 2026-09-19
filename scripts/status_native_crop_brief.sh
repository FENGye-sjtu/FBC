#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh

python - <<'PY'
from collections import deque
import json
from pathlib import Path
import subprocess

root=Path('/data_cinema/zqc/VBC')
names=('native_crop_gqa_2b_v0_contract','native_crop_clevr_2b_v0_contract')
processes=[]
for line in subprocess.check_output(['ps','-eo','pid,ppid,etime,args'],text=True).splitlines():
    if any(name in line for name in names):processes.append(line)
print('PROCESSES')
print('\n'.join(processes) if processes else 'none')
for name in names:
    path=root/'runs'/name/'summary.json'
    if not path.exists():
        print(name,'MISSING')
        continue
    value=json.loads(path.read_text())
    print(name,json.dumps({key:value.get(key) for key in (
        'count','visual_atoms','native_correct','audit_correct','direct_correct',
        'step_only_correct','efa_correct','verified','verified_correct',
        'verified_wrong','generated_steps','audit_backend','evaluation_settings')},sort_keys=True))
log=root/'logs/native_crop_contract_2b_v0.log'
print('LOG_TAIL')
if log.exists():
    with log.open(errors='replace') as stream:
        for line in deque(stream,maxlen=12):print(line.rstrip()[:2000])
else:print('missing')
PY
