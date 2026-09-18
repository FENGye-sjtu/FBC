#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
python - <<'PY'
from pathlib import Path
import inspect
import llamafactory.model.model_utils.quantization as q
path=Path(inspect.getfile(q))
print('QUANTIZATION_SOURCE',path)
print('SIGNATURE',inspect.signature(q.configure_quantization))
import llamafactory.model.patcher as patcher
print('PATCHER_USES_SAME_FUNCTION',patcher.configure_quantization is q.configure_quantization)
from llamafactory.train.tuner import run_exp
print('TRAIN_SIGNATURE',inspect.signature(run_exp))
lines=path.read_text().splitlines()
selected=set()
for i,line in enumerate(lines):
    if any(word in line for word in ['skip_modules','visual','BitsAndBytesConfig','get_forbidden_modules']):
        selected.update(range(max(0,i-5),min(len(lines),i+12)))
for i in sorted(selected):print(f'{i+1}: {lines[i]}')
PY
