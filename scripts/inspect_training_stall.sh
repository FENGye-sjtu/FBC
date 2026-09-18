#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
python - <<'PY'
from pathlib import Path
import subprocess
import json
run='/data_cinema/zqc/VBC/runs/region_sft_2b_v1.yaml'
listing=subprocess.check_output(['ps','-eo','pid,args'],text=True)
pids=[line.split()[0] for line in listing.splitlines() if run in line and 'llamafactory-cli train' in line]
if pids:
    print(subprocess.check_output(['ps','-p',','.join(pids),'-o','pid,ppid,stat,wchan:32,etime,pcpu,pmem,args'],text=True))
else:print('No live matching trainer')
paths=['/proc/pressure/io','/proc/pressure/memory']+[f'/proc/{pid}/{name}' for pid in pids for name in ['wchan','io']]
for name in paths:
    try:print(name,Path(name).read_text().strip())
    except (OSError,PermissionError) as error:print(name,type(error).__name__)
for path in Path('/data_cinema/zqc/VBC/runs/region_sft_2b_v1').glob('checkpoint-*/trainer_state.json'):
    state=json.loads(path.read_text())
    files={name:(path.parent/name).stat().st_size for name in ['adapter_model.safetensors','optimizer.pt'] if (path.parent/name).exists()}
    print('checkpoint',path.parent.name,'step',state['global_step'],files)
for pid in pids:
    try:files=list(Path(f'/proc/{pid}/fd').iterdir())
    except OSError:continue
    for fd in files:
        try:
            target=str(fd.readlink())
            if target.startswith('/data_cinema/zqc/VBC'):
                print('project_fd',fd.name,target)
        except OSError:pass
PY
