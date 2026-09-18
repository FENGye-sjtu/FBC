#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=7500 {print $1,$2}' | sort -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then echo 'WAIT_FOR_7500_MIB_FREE'; exit 3; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - <<'PY'
import json
from pathlib import Path
import yaml
from efa.export import export
root=Path('/data_cinema/zqc/VBC')
out=root/'data/derived/gqa_sft_2b_v0'
if out.exists():raise ValueError('Choose a new export version')
export(root/'data/derived/gqa_native_review_v1',out,True)
manifest=json.loads((out/'export_manifest.json').read_text())
if min(manifest['counts'][k] for k in ['efa_steps','efa_reads','efa_regions'])<1:
    raise ValueError('No reviewed training episodes')
config=yaml.safe_load((root/'code/configs/sft_3b.yaml').read_text())
config.update(model_name_or_path=str(root/'models/Qwen2-VL-2B-Instruct'),
    dataset='efa_steps,efa_reads,efa_regions',dataset_dir=str(out),
    output_dir=str(root/'runs/gqa_sft_2b_v0'),max_steps=128,
    image_max_pixels=50176,cutoff_len=1024,lora_rank=16,lora_alpha=32,
    lora_target='q_proj,k_proj,v_proj,o_proj',gradient_accumulation_steps=2,
    learning_rate=2e-5,save_steps=64)
if Path(config['output_dir']).exists():raise ValueError('Preserve existing training run')
(root/'runs/gqa_sft_2b_v0.yaml').write_text(yaml.safe_dump(config))
print('SOURCE_REVIEWED_TRAINING',json.dumps(manifest),flush=True)
PY
llamafactory-cli train "$EFA_ROOT/runs/gqa_sft_2b_v0.yaml"
python -m efa.train_latent --model "$EFA_ROOT/models/Qwen2-VL-2B-Instruct" --adapter "$EFA_ROOT/runs/gqa_sft_2b_v0" --root "$EFA_ROOT/data/derived/gqa_native_review_v1" --episode-file teacher_reviewed_episodes.jsonl --output "$EFA_ROOT/runs/latent_gqa_2b_v0" --steps 256 --engineering-only
