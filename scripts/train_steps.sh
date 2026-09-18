#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=7500 {print $1,$2}' | sort -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then exit 3; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m efa.export --root "$EFA_ROOT/data/derived/zwz_seed_v0" --output "$EFA_ROOT/data/derived/step_sft_v0" --engineering-only
python - <<'PY'
import yaml
from pathlib import Path
root=Path('/data_cinema/zqc/VBC')
config=yaml.safe_load((root/'code/configs/sft_3b.yaml').read_text())
config.update(dataset='efa_atoms',dataset_dir=str(root/'data/derived/step_sft_v0'),
    adapter_name_or_path=str(root/'runs/reader_sft_smoke_v1'),
    output_dir=str(root/'runs/step_sft_smoke_v0'),max_steps=40,
    image_max_pixels=50176,cutoff_len=768,lora_rank=8,lora_alpha=16,
    lora_target='q_proj,v_proj',gradient_accumulation_steps=1,learning_rate=1e-4)
if Path(config['output_dir']).exists():
    raise ValueError('Preserve existing run')
(root/'runs/step_sft_smoke.yaml').write_text(yaml.safe_dump(config))
PY
llamafactory-cli train "$EFA_ROOT/runs/step_sft_smoke.yaml"
python -m efa.train_latent --model "$EFA_ROOT/models/Qwen2.5-VL-3B-Instruct" --adapter "$EFA_ROOT/runs/step_sft_smoke_v0" --root "$EFA_ROOT/data/derived/zwz_seed_v0" --output "$EFA_ROOT/runs/latent_reader_smoke_v1" --steps 40 --engineering-only
