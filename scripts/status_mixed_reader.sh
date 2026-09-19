#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
python - <<'PY'
from collections import deque
import json
from pathlib import Path
import subprocess

root=Path('/data_cinema/zqc/VBC')
run=root/'runs/latent_mixed_programs_2b_v0'
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
                               '--format=csv,noheader'],text=True))
live=[]
for line in subprocess.check_output(['ps','-eo','pid,ppid,etime,args'],text=True).splitlines():
    if '-m efa.train_latent' in line and str(run) in line:
        live.append(int(line.split()[0]))
        print('LIVE_PROCESS',line)
for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                                      '--format=csv,noheader'],text=True).splitlines():
    if line.split(',')[0].strip() in {str(pid) for pid in live}:
        print('PROCESS_GPU_MEMORY',line)
metrics_path=run/'metrics.json'
metrics=json.loads(metrics_path.read_text()) if metrics_path.exists() else None
checkpoint=run/'resampler.pt'
complete=bool(metrics and metrics.get('steps')==1660 and metrics.get('examples')==830
              and checkpoint.is_file() and checkpoint.stat().st_size>0)
status='RUNNING' if live else ('COMPLETE' if complete else ('STOPPED_INCOMPLETE' if run.exists() else 'NOT_RUNNING_NO_OUTPUT'))
print(json.dumps(dict(status=status,pids=live,output_exists=run.exists(),complete_artifacts=complete)))
log=root/'logs/latent_mixed_programs_2b_v0.log'
if log.exists():
    with log.open(errors='replace') as stream:
        tail=list(deque(stream,maxlen=3))
    for line in tail:
        try:
            record=json.loads(line)
            if 'losses' in record:
                record={key:record[key] for key in ['steps','examples','episodes','seconds','peak_allocated_bytes'] if key in record}
            print('LOG_TAIL',json.dumps(record))
        except json.JSONDecodeError:
            print('LOG_TAIL',line.rstrip()[:1000])
else:
    print('LOG_MISSING',str(log))
if metrics:
    print('METRICS',json.dumps({key:metrics[key] for key in [
        'steps','examples','episodes','seconds','peak_allocated_bytes','seed','sampling','scope','calibrated'] if key in metrics}))
PY
