#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
ps -eo pid,etime,args | awk '/VBC.*reader_diagnostic_2b|python -m efa.diagnose_reader/ && !/awk/'
python - <<'PY'
from pathlib import Path
root=Path('/data_cinema/zqc/VBC')
log=root/'logs/reader_diagnostic_2b_v0.log'
if log.exists():print('\n'.join(log.read_text(errors='replace').splitlines()[-5:]))
summary=root/'runs/reader_diagnostic_2b_v0/summary.json'
print(summary.read_text() if summary.exists() else 'SUMMARY_PENDING')
PY
