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
    if any(s in line for s in ['python -m efa.sft_entry','tools/continue_mixed_programs_v0.sh']):print(line)
for name in ['logs/continue_mixed_programs_v0.log','runs/mixed_program_sft_2b_v0/trainer_log.jsonl']:
    path=root/name
    if path.exists():print(name,'\n'.join(path.read_text(errors='replace').splitlines()[-3:])[-2500:])
for name in ['runs/mixed_program_sft_2b_v0/precision_policy.json','runs/mixed_program_sft_2b_v0/all_results.json']:
    path=root/name
    print(name,path.read_text() if path.exists() else 'PENDING')
PY
