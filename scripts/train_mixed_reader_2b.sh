#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
python - <<'PY'
import json
import math
from pathlib import Path
from efa.reader_data import load_training_sources

root=Path('/data_cinema/zqc/VBC')
adapter=root/'runs/mixed_program_sft_2b_v0'
output=root/'runs/latent_mixed_programs_2b_v0'
if output.exists():raise ValueError('Preserve existing mixed-reader run; choose a new version')
for name in ['adapter_model.safetensors','adapter_config.json','all_results.json','trainer_state.json','precision_policy.json']:
    path=adapter/name
    if not path.is_file() or not path.stat().st_size:raise ValueError('Missing SFT artifact: '+name)
state=json.loads((adapter/'trainer_state.json').read_text())
if state.get('global_step')!=2400 or state.get('max_steps')!=2400:
    raise ValueError('The mixed SFT must finish all 2400 optimizer updates')
metrics=json.loads((adapter/'all_results.json').read_text())
if not math.isfinite(metrics.get('train_loss',float('nan'))) or metrics.get('train_runtime',0)<=0:
    raise ValueError('Completed SFT training metrics are required')
precision=json.loads((adapter/'precision_policy.json').read_text())
if precision.get('mode')!='bf16_vision_nf4_language':
    raise ValueError('Expected completed BF16-vision/NF4-language SFT')
rows,manifest=load_training_sources(
    root/'data/derived/gqa_native_review_v2_complete','teacher_reviewed_episodes.jsonl',
    [(root/'data/derived/gqa_native_long_review_v1_completed63','teacher_reviewed_episodes.jsonl'),
     (root/'data/derived/clevr_count_program_v1','native_episodes.jsonl')],engineering_only=True)
if len(rows)!=830 or manifest['episodes']!=344:
    raise ValueError('The pinned old-data comparison requires 830 visual atoms from 344 episodes')
if [source['eligible_visual_atoms'] for source in manifest['sources']]!=[384,20,426]:
    raise ValueError('Unexpected source composition for the old-data comparison')
print(json.dumps(dict(scope='old_native_and_teacher_data_positive_alignment_comparator',
    newly_codex_adjudicated=False,training_steps=1660,epochs=2,sft_global_step=state['global_step'],
    input_sources=manifest)),flush=True)
PY
# Prior 2B reader training used about 3 GiB. Require 6 GiB free at launch,
# preserving additional headroom for the shared GPUs; never stop another job.
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=6000 {print $1,$2}' | sort -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then echo 'WAIT_FOR_6000_MIB_FREE'; exit 3; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -i "$gpu" --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader
exec python -m efa.train_latent \
    --model "$EFA_ROOT/models/Qwen2-VL-2B-Instruct" \
    --adapter "$EFA_ROOT/runs/mixed_program_sft_2b_v0" \
    --root "$EFA_ROOT/data/derived/gqa_native_review_v2_complete" \
    --episode-file teacher_reviewed_episodes.jsonl \
    --episode-source "$EFA_ROOT/data/derived/gqa_native_long_review_v1_completed63" teacher_reviewed_episodes.jsonl \
    --episode-source "$EFA_ROOT/data/derived/clevr_count_program_v1" native_episodes.jsonl \
    --output "$EFA_ROOT/runs/latent_mixed_programs_2b_v0" \
    --steps 1660 --seed 2027 --engineering-only
