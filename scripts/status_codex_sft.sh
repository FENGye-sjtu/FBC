#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
python - <<'PY'
from collections import deque
import json
from pathlib import Path
import subprocess

root=Path('/data_cinema/zqc/VBC');run=root/'runs/codex_sft_2b_v0'
config=root/'runs/codex_sft_2b_v0.yaml'
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
                               '--format=csv,noheader'],text=True))
live=[]
for line in subprocess.check_output(['ps','-eo','pid,ppid,etime,args'],text=True).splitlines():
    training='-m efa.sft_entry' in line and str(config) in line
    launcher=str(root/'tools/codex_sft_2b_v0.sh') in line
    if training or launcher:
        live.append(int(line.split()[0]));print('LIVE_PROCESS',line)
for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                                      '--format=csv,noheader'],text=True).splitlines():
    if line.split(',')[0].strip() in {str(pid) for pid in live}:print('PROCESS_GPU_MEMORY',line)

def read_json(path):
    if not path.exists():return None
    try:return json.loads(path.read_text())
    except json.JSONDecodeError:return None

manifest=read_json(root/'data/derived/codex_sft_2b_v0/export_manifest.json')
if manifest:
    print('DATA',json.dumps({key:manifest[key] for key in ['train_episodes','train_images','counts',
        'step_sampling_multiplier','weighted_samples','full_epochs','expected_optimizer_updates','scope']}))
for path in [root/'logs/codex_sft_2b_v0.log',run/'trainer_log.jsonl']:
    if path.exists():
        with path.open(errors='replace') as stream:tail=list(deque(stream,maxlen=3))
        for line in tail:print('LOG_TAIL',str(path.relative_to(root)),line.rstrip()[:1600])
    else:print('LOG_MISSING',str(path.relative_to(root)))
state=read_json(run/'trainer_state.json')
if state is None:
    checkpoints=sorted(run.glob('checkpoint-*/trainer_state.json'),
                       key=lambda path:int(path.parent.name.removeprefix('checkpoint-')),reverse=True)
    for checkpoint in checkpoints:
        state=read_json(checkpoint)
        if state is not None:break
if state:print('TRAINER_STATE',json.dumps({key:state.get(key) for key in ['global_step','max_steps','epoch']}))
metrics=read_json(run/'all_results.json')
precision=read_json(run/'precision_policy.json')
if metrics:print('METRICS',json.dumps(metrics))
if precision:print('PRECISION',json.dumps(precision))
weights=run/'adapter_model.safetensors'
complete=bool(manifest and state and metrics and weights.is_file() and weights.stat().st_size>0
              and state.get('global_step')==manifest['expected_optimizer_updates']
              and abs(state.get('epoch',0)-2.0)<=1e-6)
print('STATUS',json.dumps(dict(pids=live,complete_artifacts=complete,
    status='RUNNING' if live else ('COMPLETE' if complete else 'NOT_RUNNING_INCOMPLETE'))))
PY
