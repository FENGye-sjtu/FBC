#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
python -m unittest discover -s tests -q
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=7000 {print $1,$2}' | sort -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then exit 3; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - <<'PY'
import yaml
from pathlib import Path
root=Path('/data_cinema/zqc/VBC')
if not (root/'models/Qwen2-VL-2B-Instruct/source_revision.json').exists():raise ValueError('Model download not complete')
config=yaml.safe_load((root/'code/configs/sft_3b.yaml').read_text())
config.update(model_name_or_path=str(root/'models/Qwen2-VL-2B-Instruct'),
    dataset='efa_atoms,efa_regions,efa_reads',dataset_dir=str(root/'data/derived/region_sft_pilot_v2'),
    output_dir=str(root/'runs/region_sft_2b_v1'),max_steps=200,
    image_max_pixels=50176,cutoff_len=512,lora_rank=8,lora_alpha=16,
    lora_target='q_proj,v_proj',gradient_accumulation_steps=4,learning_rate=5e-5)
if Path(config['output_dir']).exists():raise ValueError('Preserve existing run')
(root/'runs/region_sft_2b_v1.yaml').write_text(yaml.safe_dump(config))
PY
llamafactory-cli train "$EFA_ROOT/runs/region_sft_2b_v1.yaml"
python -m efa.train_latent --model "$EFA_ROOT/models/Qwen2-VL-2B-Instruct" --adapter "$EFA_ROOT/runs/region_sft_2b_v1" --root "$EFA_ROOT/data/derived/zwz_pilot_v1" --output "$EFA_ROOT/runs/latent_reader_2b_v1" --steps 256 --engineering-only
