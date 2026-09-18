#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=7000 {print $1,$2}' | sort -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then exit 3; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - <<'PY'
import json
from pathlib import Path
import yaml
from efa.export import export
root=Path('/data_cinema/zqc/VBC');out=root/'data/derived/episode_sft_2b_v0'
if out.exists():raise ValueError('Preserve the previous episode export')
out.mkdir()
merged={name:[] for name in ['efa_steps','efa_reads','efa_regions']}
for source in ['viscot_gqa_plus_v2','chartqa_plus_v2']:
    part=out/source
    export(root/'data/derived'/source,part,True)
    for name in merged:merged[name].extend(json.loads((part/(name+'.json')).read_text()))
info=json.loads((part/'dataset_info.json').read_text())
for name,rows in merged.items():
    if not rows:raise ValueError('No eligible episode supervision')
    (out/(name+'.json')).write_text(json.dumps(rows))
(out/'dataset_info.json').write_text(json.dumps({k:info[k] for k in merged}))
(out/'export_manifest.json').write_text(json.dumps(dict(counts={k:len(v) for k,v in merged.items()},engineering_only=True)))
config=yaml.safe_load((root/'code/configs/sft_3b.yaml').read_text())
config.update(model_name_or_path=str(root/'models/Qwen2-VL-2B-Instruct'),
    adapter_name_or_path=str(root/'runs/region_sft_2b_v1'),dataset=','.join(merged),dataset_dir=str(out),
    output_dir=str(root/'runs/episode_sft_2b_v0'),max_steps=32,
    image_max_pixels=50176,cutoff_len=1024,lora_rank=8,lora_alpha=16,
    lora_target='q_proj,v_proj',gradient_accumulation_steps=1,learning_rate=5e-5)
if Path(config['output_dir']).exists():raise ValueError('Preserve existing run')
(root/'runs/episode_sft_2b_v0.yaml').write_text(yaml.safe_dump(config))
print('EPISODE_TRAIN_COUNTS',json.dumps({k:len(v) for k,v in merged.items()}),flush=True)
PY
llamafactory-cli train "$EFA_ROOT/runs/episode_sft_2b_v0.yaml"
python -m efa.train_latent --model "$EFA_ROOT/models/Qwen2-VL-2B-Instruct" --adapter "$EFA_ROOT/runs/episode_sft_2b_v0" --resampler "$EFA_ROOT/runs/latent_reader_2b_v1/resampler.pt" --root "$EFA_ROOT/data/derived/zwz_pilot_v1" --output "$EFA_ROOT/runs/latent_episode_2b_v0" --steps 64 --engineering-only
