#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
python -u - <<'PY'
import json
from pathlib import Path
import re
import subprocess
import time
root=Path('/data_cinema/zqc/VBC');idle=0
for tick in range(50):
    log=root/'logs/gqa_mixed_training_2b_v2.log'
    lines=log.read_text(errors='replace').splitlines()
    progress=[line for line in lines if re.search(r'\d+/1200',line) or line.startswith('{"step":')]
    summary=root/'runs/gqa_heldout_2b_v2/summary.json'
    stages={name:(root/'runs'/name/file).exists() for name,file in [
        ('gqa_sft_2b_v2','all_results.json'),('latent_gqa_2b_v2','metrics.json'),('gqa_heldout_2b_v2','summary.json')]}
    commands=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
    active=any('python -m efa.' in line and any('/runs/'+name in line for name in stages) for line in commands)
    print(json.dumps(dict(stages=stages,active=active,last_progress=progress[-1][:250] if progress else None)),flush=True)
    if summary.exists():
        print(summary.read_text(),flush=True);break
    idle=0 if active else idle+1
    if idle>=2:
        print('\n'.join(lines[-20:]),flush=True);raise SystemExit('Pipeline stopped before evaluation completion')
    time.sleep(45)
PY
