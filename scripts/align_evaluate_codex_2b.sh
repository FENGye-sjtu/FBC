#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
python - <<'PY'
import hashlib
import json
import math
from pathlib import Path
from efa.reader_data import load_training_sources
from efa.evaluate_episodes import select_episodes

root=Path('/data_cinema/zqc/VBC')
adapter=root/'runs/codex_sft_2b_v0'
initial=root/'runs/latent_mixed_programs_2b_v0'
for name in ['latent_codex_sft_2b_v0','codex_gqa_heldout_2b_v0','codex_clevr_heldout_2b_v0']:
    if (root/'runs'/name).exists():raise ValueError('Preserve existing run: '+name)
for path in [adapter/'adapter_model.safetensors',adapter/'adapter_config.json',initial/'resampler.pt']:
    if not path.is_file() or not path.stat().st_size:raise ValueError('Missing model artifact: '+str(path))
state=json.loads((adapter/'trainer_state.json').read_text())
if state.get('global_step')!=506 or state.get('max_steps')!=506 or abs(state.get('epoch',0)-2.0)>1e-6:
    raise ValueError('Codex SFT must finish 506 updates and two complete epochs first')
metrics=json.loads((adapter/'all_results.json').read_text())
if not math.isfinite(metrics.get('train_loss',float('nan'))) or metrics.get('train_runtime',0)<=0:
    raise ValueError('Completed Codex SFT metrics are required')
if json.loads((adapter/'precision_policy.json').read_text()).get('mode')!='bf16_vision_nf4_language':
    raise ValueError('Codex SFT must use BF16 vision and NF4 language')
previous=json.loads((initial/'metrics.json').read_text())
if previous.get('steps')!=1660 or previous.get('examples')!=830:
    raise ValueError('The initial mixed reader must be the completed 1660-update run')
rows,manifest=load_training_sources(root/'data/derived/codex_adjudicated_v2','reviewed_episodes.jsonl',
    [(root/'data/derived/codex_curated45_adjudicated_v1','reviewed_episodes.jsonl')],engineering_only=True)
if len(rows)!=69 or manifest['episodes']!=50 or manifest['images']!=50:
    raise ValueError('Expected 69 reviewed visual atoms from 50 training episodes/images')
if [source['eligible_visual_atoms'] for source in manifest['sources']]!=[16,53]:
    raise ValueError('Unexpected Codex source composition')
if any(row['annotation_status']!='reviewed_episode' or row.get('label_basis')!=
       'codex_model_visual_adjudication_not_human_gold_or_runtime_audit' for row in rows):
    raise ValueError('Only Codex-adjudicated visual atoms may enter alignment')
train_images={row['image_sha256'] for row in rows}
cases=[('gqa_native_review_v2_complete','teacher_reviewed_episodes.jsonl',
        ['gqa-native-9236','gqa-native-28788','gqa-native-94674']),
       ('clevr_count_program_v1','native_episodes.jsonl',
        ['clevr-count-10563','clevr-count-263129','clevr-count-231696'])]
for source,tier,expected in cases:
    selected=select_episodes(root/'data/derived'/source,tier,'dev',3,engineering_only=True)
    if [row['id'] for row in selected]!=expected:raise ValueError('Held-out selection changed: '+source)
    if any(row['image_sha256'] in train_images for row in selected):raise ValueError('Codex training image reached held-out evaluation')
policy=root/'code/configs/policy_engineering.json'
print(json.dumps(dict(scope='codex_positive_reader_realignment_then_fixed_development_comparison',
    adapter_sha256=hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),
    initial_resampler=str(initial/'resampler.pt'),
    initial_resampler_sha256=hashlib.sha256((initial/'resampler.pt').read_bytes()).hexdigest(),
    initial_reader_for_training_only=True,alignment_steps=276,alignment_passes=4,
    input_sources=manifest,policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),calibrated=False)),flush=True)
PY
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=6500 {print $1,$2}' | sort -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then echo 'WAIT_FOR_6500_MIB_FREE'; exit 3; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -i "$gpu" --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader
# The old reader is permitted only as a training initialization. This command
# updates it with the new shared adapter and writes a newly bound checkpoint.
python -m efa.train_latent \
    --model "$EFA_ROOT/models/Qwen2-VL-2B-Instruct" \
    --adapter "$EFA_ROOT/runs/codex_sft_2b_v0" \
    --resampler "$EFA_ROOT/runs/latent_mixed_programs_2b_v0/resampler.pt" \
    --root "$EFA_ROOT/data/derived/codex_adjudicated_v2" \
    --episode-file reviewed_episodes.jsonl \
    --episode-source "$EFA_ROOT/data/derived/codex_curated45_adjudicated_v1" reviewed_episodes.jsonl \
    --output "$EFA_ROOT/runs/latent_codex_sft_2b_v0" \
    --steps 276 --seed 2027 --engineering-only
python - <<'PY'
import hashlib
import json
from pathlib import Path
root=Path('/data_cinema/zqc/VBC')
reader=root/'runs/latent_codex_sft_2b_v0'
metrics=json.loads((reader/'metrics.json').read_text())
if (metrics.get('steps'),metrics.get('examples'),metrics.get('episodes'))!=(276,69,50):
    raise ValueError('Reader realignment is incomplete')
weights=reader/'resampler.pt'
if not weights.is_file() or not weights.stat().st_size:raise ValueError('Missing newly aligned reader')
adapter_sha=hashlib.sha256((root/'runs/codex_sft_2b_v0/adapter_model.safetensors').read_bytes()).hexdigest()
if metrics['encoder_config'].get('adapter_sha256')!=adapter_sha or metrics['encoder_config'].get('vision_quantized') is not False:
    raise ValueError('New reader metadata must match the Codex adapter and BF16 vision')
print(json.dumps(dict(reader_ready_for_evaluation=True,steps=276,examples=69,episodes=50,
    adapter_sha256=adapter_sha,resampler_sha256=hashlib.sha256(weights.read_bytes()).hexdigest())),flush=True)
# The inference backend additionally validates the checkpoint's own adapter hash.
PY
codex_eval_args=(
    --model "$EFA_ROOT/models/Qwen2-VL-2B-Instruct"
    --adapter "$EFA_ROOT/runs/codex_sft_2b_v0"
    --resampler "$EFA_ROOT/runs/latent_codex_sft_2b_v0/resampler.pt"
    --policy "$EFA_ROOT/code/configs/policy_engineering.json"
    --split dev --limit 3 --engineering-only
)
python -m efa.evaluate_episodes "${codex_eval_args[@]}" \
    --root "$EFA_ROOT/data/derived/gqa_native_review_v2_complete" \
    --episode-file teacher_reviewed_episodes.jsonl \
    --output "$EFA_ROOT/runs/codex_gqa_heldout_2b_v0"
exec python -m efa.evaluate_episodes "${codex_eval_args[@]}" \
    --root "$EFA_ROOT/data/derived/clevr_count_program_v1" \
    --episode-file native_episodes.jsonl \
    --output "$EFA_ROOT/runs/codex_clevr_heldout_2b_v0"
