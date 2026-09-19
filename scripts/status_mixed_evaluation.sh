#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
python - <<'PY'
from collections import deque
import json
from pathlib import Path
import subprocess

root=Path('/data_cinema/zqc/VBC')
names=['mixed_gqa_heldout_2b_v0','mixed_clevr_heldout_2b_v0']
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
                               '--format=csv,noheader'],text=True))
live=[]
for line in subprocess.check_output(['ps','-eo','pid,ppid,etime,args'],text=True).splitlines():
    evaluation='-m efa.evaluate_episodes' in line and any(str(root/'runs'/name) in line for name in names)
    launcher=str(root/'tools/mixed_programs_eval_v0.sh') in line
    if evaluation or launcher:
        live.append(int(line.split()[0]))
        print('LIVE_PROCESS',line)
live_ids={str(pid) for pid in live}
for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                                      '--format=csv,noheader'],text=True).splitlines():
    if line.split(',')[0].strip() in live_ids:print('PROCESS_GPU_MEMORY',line)

def compact(value):
    if isinstance(value,dict):return {key:compact(item) for key,item in value.items() if key not in {'calls','checkpoint'}}
    if isinstance(value,list):return [compact(item) for item in value]
    return value

completed=0
for name in names:
    out=root/'runs'/name
    results=out/'results.jsonl'
    rows=[];incomplete_lines=0
    if results.exists():
        for line in results.read_text(errors='replace').splitlines():
            if not line.strip():continue
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:incomplete_lines+=1
    summary_path=out/'summary.json'
    summary=None
    if summary_path.exists():
        try:summary=json.loads(summary_path.read_text())
        except json.JSONDecodeError:pass
    complete=bool(summary and summary.get('count')==3 and len(rows)==3 and not incomplete_lines)
    completed+=complete
    print('RUN',json.dumps(dict(name=name,output_exists=out.exists(),result_rows=len(rows),
        incomplete_result_lines=incomplete_lines,summary_exists=summary_path.exists(),complete=complete)))
    for row in rows:
        print('RESULT',json.dumps(dict(id=row['id'],reference=row['reference'],
            native_correct=sum(item['native_correct'] for item in row['reads']),
            compact_correct=sum(item['compact_correct'] for item in row['reads']),visual_atoms=len(row['reads']),
            premise_correct=sum(item['correct'] for item in row.get('reference_premise_reads',[])),
            premise_atoms=len(row.get('reference_premise_reads',[])),
            direct=compact(row['direct']),step_only=compact(row['step_only']),efa=compact(row['efa']))))
    if summary:print('SUMMARY',json.dumps(compact(summary)))
print('STATUS',json.dumps(dict(pids=live,completed_runs=completed,
    status='RUNNING' if live else ('COMPLETE' if completed==2 else 'NOT_RUNNING_INCOMPLETE'))))
log=root/'logs/mixed_programs_eval_v0.log'
if log.exists():
    with log.open(errors='replace') as stream:tail=list(deque(stream,maxlen=4))
    for line in tail:
        try:print('LOG_TAIL',json.dumps(compact(json.loads(line))))
        except json.JSONDecodeError:print('LOG_TAIL',line.rstrip()[:1200])
else:print('LOG_MISSING',str(log))
PY
