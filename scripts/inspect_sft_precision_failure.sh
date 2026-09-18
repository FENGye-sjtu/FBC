#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
python - <<'PY'
from pathlib import Path
root=Path('/data_cinema/zqc/VBC')
log=root/'logs/gqa_mixed_training_2b_v1_launch2.log'
for line in log.read_text(errors='replace').splitlines():
    if any(s in line.lower() for s in ['quantiz','4bit','trainable params','dtype','freez','precision','error']):
        print(line[:1200])
print('\n'.join(log.read_text(errors='replace').splitlines()[-28:]))
import inspect
from llamafactory.model.model_utils.quantization import configure_quantization
from llamafactory.extras.constants import QuantizationMethod
print('QUANTIZATION_METHODS', [(q.name,q.value) for q in QuantizationMethod])
import yaml
config=yaml.safe_load((root/'runs/gqa_sft_2b_v1.yaml').read_text())
print('REQUESTED_METHOD',config['quantization_method'])
PY
