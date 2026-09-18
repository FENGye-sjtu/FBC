#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
python -m efa.export --root "$EFA_ROOT/data/derived/zwz_seed_v0" --output "$EFA_ROOT/data/derived/sft_v0" --engineering-only
# The initial region-reader smoke is not represented as full episode training.
python - <<'PY'
import json,yaml
from pathlib import Path
root=Path('/data_cinema/zqc/VBC')
config=yaml.safe_load((root/'code/configs/sft_3b.yaml').read_text())
manifest=json.loads((root/'data/derived/sft_v0/export_manifest.json').read_text())
names=[k for k,v in manifest['counts'].items() if v]
if not names:
    raise RuntimeError('No usable SFT records')
config['dataset']=','.join(names)
config['max_steps']=10
config['output_dir']=str(root/'runs/reader_sft_smoke_v1')
config['image_max_pixels']=50176
config['cutoff_len']=512
config['lora_rank']=8
config['lora_alpha']=16
config['lora_target']='q_proj,v_proj'
config['gradient_accumulation_steps']=4
if Path(config['output_dir']).exists():
    raise RuntimeError('Preserve previous run; choose a new run ID')
(root/'runs/reader_sft_smoke.yaml').write_text(yaml.safe_dump(config))
PY
# This engineering allocation needs free memory; do not terminate other jobs.
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=7500 {print $1,$2}' | sort -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then
  printf 'WAITING_FOR_GPU: requires at least 7500 MiB free for quantized smoke\n'
  exit 3
fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
llamafactory-cli train "$EFA_ROOT/runs/reader_sft_smoke.yaml"
