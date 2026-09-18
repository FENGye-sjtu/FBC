#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
python - <<'PY'
import json
from pathlib import Path
root=Path('/data_cinema/zqc/VBC')
c=json.loads((root/'models/Qwen2-VL-2B-Instruct/config.json').read_text())
print('tie_word_embeddings',c.get('tie_word_embeddings'))
print('\n'.join((root/'logs/vision_precision_2b_v1.log').read_text().splitlines()[-45:]))
PY
