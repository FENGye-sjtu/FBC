#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
test -f "$EFA_ROOT/runs/gqa_heldout_2b_v2/summary.json"
test -f "$EFA_ROOT/runs/gqa_reader_objectives_eval_v1/summary.json"
test -f "$EFA_ROOT/data/derived/gqa_native_long_review_v1_completed63/manifest.json"
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=7000 {print $1,$2}' | sort -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then echo 'WAIT_FOR_7000_MIB_FREE'; exit 3; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - <<'PY'
import json
from pathlib import Path
import yaml
from efa.mix_sft import mix
root=Path('/data_cinema/zqc/VBC');out=root/'data/derived/mixed_program_sft_2b_v0'
long=root/'data/derived/gqa_native_long_review_v1_completed63'
review=json.loads((long/'manifest.json').read_text())
if not review['construction_complete'] or review.get('errors',0) or not review['approved_splits'].get('train'):
    raise ValueError('Finish long-chain source review before training')
manifest=mix([root/'data/derived/gqa_native_review_v2_complete',root/'data/derived/clevr_count_program_v1',long],out,True)
steps=json.loads((out/'efa_steps.json').read_text())
(out/'efa_steps_weighted.json').write_text(json.dumps(steps*4))
info=json.loads((out/'dataset_info.json').read_text())
info['efa_steps_weighted']=dict(info['efa_steps'],file_name='efa_steps_weighted.json')
(out/'dataset_info.json').write_text(json.dumps(info))
manifest.update(step_sampling_multiplier=4,unique_step_examples=len(steps))
(out/'export_manifest.json').write_text(json.dumps(manifest,indent=2))
config=yaml.safe_load((root/'code/configs/sft_3b.yaml').read_text())
config.update(model_name_or_path=str(root/'models/Qwen2-VL-2B-Instruct'),
    dataset='efa_steps_weighted,efa_reads,efa_regions,efa_premises,efa_bindings',dataset_dir=str(out),
    output_dir=str(root/'runs/mixed_program_sft_2b_v0'),max_steps=2400,
    image_max_pixels=50176,cutoff_len=1024,lora_rank=16,lora_alpha=32,
    lora_target='q_proj,k_proj,v_proj,o_proj',gradient_accumulation_steps=2,
    learning_rate=5e-5,save_steps=400)
if Path(config['output_dir']).exists():raise ValueError('Preserve existing model run')
(root/'runs/mixed_program_sft_2b_v0.yaml').write_text(yaml.safe_dump(config))
print(json.dumps(manifest),flush=True)
PY
python -m efa.sft_entry "$EFA_ROOT/runs/mixed_program_sft_2b_v0.yaml"
