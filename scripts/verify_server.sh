#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
python -m unittest discover -s tests -v
python - <<'PY'
import json
from pathlib import Path
root=Path('/data_cinema/zqc/VBC')
for name in ['data/derived/sft_v0/export_manifest.json',
             'runs/reader_sft_smoke_v1/all_results.json',
             'runs/reader_sft_smoke_v1/trainer_state.json',
             'runs/model_smoke.json','runs/latent_reader_smoke_v0/metrics.json',
             'runs/episode_smoke_v0.json','runs/episode_smoke_v1.json',
             'runs/step_sft_smoke_v0/all_results.json','runs/latent_reader_smoke_v1/metrics.json',
             'data/derived/zwz_pilot_v1/manifest.json']:
    path=root/name
    if not path.exists():
        print(name,'PENDING');continue
    value=json.loads(path.read_text())
    if name.endswith('trainer_state.json'):
        value={k:value[k] for k in ['global_step','max_steps','epoch']}
    elif name.endswith('model_smoke.json'):
        value={k:value[k] for k in ['task','generated','reference','scores','peak_allocated_bytes']}
    elif name.endswith('metrics.json'):
        value={k:v for k,v in value.items() if k!='losses'}
    elif 'episode_smoke_' in name:
        value={k:value[k] for k in ['final','budget','implementation']}
    print(name,json.dumps(value))
adapter=root/'runs/reader_sft_smoke_v1/adapter_model.safetensors'
print('adapter_bytes',adapter.stat().st_size if adapter.exists() else 'PENDING')
PY
for label in step_train zwz_pilot episode_smoke_v1 deepeyesv2_data; do
  printf '\n%s\n' "$label"
  tail -n 8 "$EFA_ROOT/logs/$label.log" 2>/dev/null || true
done
