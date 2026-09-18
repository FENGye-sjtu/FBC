#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
ps -eo pid,etime,args | awk '/VBC.*gqa_heldout_2b|python -m efa.evaluate_episodes/ && !/awk/'
python - <<'PY'
from pathlib import Path
root=Path('/data_cinema/zqc/VBC')
log=root/'logs/gqa_heldout_2b_v0.log'
if log.exists():print('\n'.join(line[:1200] for line in log.read_text(errors='replace').splitlines()[-8:]))
summary=root/'runs/gqa_heldout_2b_v0/summary.json'
print(summary.read_text() if summary.exists() else 'SUMMARY_PENDING')
PY
