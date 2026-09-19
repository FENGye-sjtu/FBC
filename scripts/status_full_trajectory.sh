#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
python - <<'PY'
from collections import deque
import json
from pathlib import Path
import subprocess

root=Path('/data_cinema/zqc/VBC')
needles=['codex_full_trajectory_2b_v0','latent_full_trajectory_2b_v0','full_evidence_2b_v0']
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
                               '--format=csv,noheader'],text=True))
for line in subprocess.check_output(['ps','-eo','pid,ppid,etime,args'],text=True).splitlines():
    if any(name in line for name in needles):print('LIVE_PROCESS',line)
for relative in ['data/derived/codex_full_trajectory_2b_v0/export_manifest.json',
                 'runs/codex_full_trajectory_2b_v0/trainer_state.json',
                 'runs/codex_full_trajectory_2b_v0/all_results.json',
                 'runs/latent_full_trajectory_2b_v0/metrics.json',
                 'runs/full_evidence_2b_v0/metrics.json',
                 'runs/full_evidence_gqa_2b_v0/summary.json',
                 'runs/full_evidence_clevr_2b_v0/summary.json']:
    path=root/relative
    if path.exists():
        try:print('ARTIFACT',relative,json.dumps(json.loads(path.read_text())))
        except json.JSONDecodeError:print('INCOMPLETE_JSON',relative)
    else:print('MISSING',relative)
for relative in ['logs/full_trajectory_2b_v0.log','logs/full_evidence_2b_v0.log']:
    path=root/relative
    if path.exists():
        with path.open(errors='replace') as stream:tail=list(deque(stream,maxlen=4))
        for line in tail:print('LOG_TAIL',relative,line.rstrip()[:1800])
PY
