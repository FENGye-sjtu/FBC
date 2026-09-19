#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
python - <<'PY'
import hashlib
import json
from pathlib import Path
from efa.evaluate_episodes import select_episodes

root=Path('/data_cinema/zqc/VBC')
reader=root/'runs/latent_mixed_programs_2b_v0'
metrics=json.loads((reader/'metrics.json').read_text())
if metrics.get('steps')!=1660 or metrics.get('examples')!=830:
    raise ValueError('Mixed reader must complete 1660 updates over 830 visual atoms')
for path in [reader/'resampler.pt',root/'runs/mixed_program_sft_2b_v0/adapter_model.safetensors']:
    if not path.is_file() or not path.stat().st_size:raise ValueError('Missing model artifact: '+str(path))
cases=[
    ('gqa_native_review_v2_complete','teacher_reviewed_episodes.jsonl','mixed_gqa_heldout_2b_v0',
     ['gqa-native-9236','gqa-native-28788','gqa-native-94674']),
    ('clevr_count_program_v1','native_episodes.jsonl','mixed_clevr_heldout_2b_v0',
     ['clevr-count-10563','clevr-count-263129','clevr-count-231696']),
]
for source,tier,output,expected in cases:
    if (root/'runs'/output).exists():raise ValueError('Preserve existing evaluation: '+output)
    rows=select_episodes(root/'data/derived'/source,tier,'dev',3,engineering_only=True)
    if [row['id'] for row in rows]!=expected:raise ValueError('Held-out selection changed: '+source)
    print(json.dumps(dict(source=source,annotation_tier=tier,ids=expected,output=output)),flush=True)
policy=root/'code/configs/policy_engineering.json'
print(json.dumps(dict(policy=str(policy),policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
    calibrated=False,reader_steps=metrics['steps'],reader_examples=metrics['examples'],
    scope='old_native_and_teacher_data_mixed_sft_comparison')),flush=True)
PY
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=6500 {print $1,$2}' | sort -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then echo 'WAIT_FOR_6500_MIB_FREE'; exit 3; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -i "$gpu" --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader
mixed_eval_args=(
    --model "$EFA_ROOT/models/Qwen2-VL-2B-Instruct"
    --adapter "$EFA_ROOT/runs/mixed_program_sft_2b_v0"
    --resampler "$EFA_ROOT/runs/latent_mixed_programs_2b_v0/resampler.pt"
    --policy "$EFA_ROOT/code/configs/policy_engineering.json"
    --split dev --limit 3 --engineering-only
)
python -m efa.evaluate_episodes "${mixed_eval_args[@]}" \
    --root "$EFA_ROOT/data/derived/gqa_native_review_v2_complete" \
    --episode-file teacher_reviewed_episodes.jsonl \
    --output "$EFA_ROOT/runs/mixed_gqa_heldout_2b_v0"
exec python -m efa.evaluate_episodes "${mixed_eval_args[@]}" \
    --root "$EFA_ROOT/data/derived/clevr_count_program_v1" \
    --episode-file native_episodes.jsonl \
    --output "$EFA_ROOT/runs/mixed_clevr_heldout_2b_v0"
