#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
python - <<'PY'
import json
from pathlib import Path
from efa.codex_objectives import load_correction_pairs,attach_reviewed_replay
from efa.reader_data import load_training_sources
root=Path('/data_cinema/zqc/VBC')
adapter=root/'runs/codex_full_trajectory_2b_v0'
state=json.loads((adapter/'trainer_state.json').read_text())
if state.get('global_step')!=114:raise ValueError('Finish complete-trajectory SFT first')
for name in ['latent_full_trajectory_2b_v0','full_evidence_2b_v0','full_evidence_gqa_2b_v0','full_evidence_clevr_2b_v0']:
    if (root/'runs'/name).exists():raise ValueError('Preserve existing run: '+name)
extra=[(root/'data/derived/codex_curated45_adjudicated_v1','reviewed_episodes.jsonl'),
       (root/'data/derived/codex_local32_adjudicated_v1','reviewed_episodes.jsonl')]
reads,manifest=load_training_sources(root/'data/derived/codex_adjudicated_v2','reviewed_episodes.jsonl',extra,True)
pairs,pair_manifest=load_correction_pairs(root/'data/derived/codex_correction_pairs_v1',True)
rows=attach_reviewed_replay(pairs,reads)
if (len(reads),manifest['episodes'],manifest['images'],len(pairs),len(rows))!=(106,75,75,16,122):
    raise ValueError('Unexpected evidence-training composition')
print(json.dumps(dict(reads=106,episodes=75,reader_updates=424,correction_pairs=16,
    correction_and_replay_updates=244,actual_student_repairs=0)),flush=True)
PY
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=8500 {print $1,$2}' | sort -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then echo 'WAIT_FOR_8500_MIB_FREE'; exit 3; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -i "$gpu" --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader
python -m efa.train_latent \
  --model "$EFA_ROOT/models/Qwen2-VL-2B-Instruct" \
  --adapter "$EFA_ROOT/runs/codex_full_trajectory_2b_v0" \
  --resampler "$EFA_ROOT/runs/latent_mixed_programs_2b_v0/resampler.pt" \
  --root "$EFA_ROOT/data/derived/codex_adjudicated_v2" \
  --episode-file reviewed_episodes.jsonl \
  --episode-source "$EFA_ROOT/data/derived/codex_curated45_adjudicated_v1" reviewed_episodes.jsonl \
  --episode-source "$EFA_ROOT/data/derived/codex_local32_adjudicated_v1" reviewed_episodes.jsonl \
  --output "$EFA_ROOT/runs/latent_full_trajectory_2b_v0" \
  --steps 424 --seed 2027 --engineering-only
python -m efa.train_codex_objectives \
  --model "$EFA_ROOT/models/Qwen2-VL-2B-Instruct" \
  --adapter "$EFA_ROOT/runs/codex_full_trajectory_2b_v0" \
  --resampler "$EFA_ROOT/runs/latent_full_trajectory_2b_v0/resampler.pt" \
  --root "$EFA_ROOT/data/derived/codex_correction_pairs_v1" \
  --replay-root "$EFA_ROOT/data/derived/codex_adjudicated_v2" \
  --replay-source "$EFA_ROOT/data/derived/codex_curated45_adjudicated_v1" \
  --replay-source "$EFA_ROOT/data/derived/codex_local32_adjudicated_v1" \
  --output "$EFA_ROOT/runs/full_evidence_2b_v0" \
  --epochs 2 --train-adapter --adapter-learning-rate 1e-6 --engineering-only
eval_args=(
  --model "$EFA_ROOT/models/Qwen2-VL-2B-Instruct"
  --adapter "$EFA_ROOT/runs/full_evidence_2b_v0/adapter"
  --resampler "$EFA_ROOT/runs/full_evidence_2b_v0/resampler.pt"
  --policy "$EFA_ROOT/code/configs/policy_engineering.json"
  --split dev --limit 3 --engineering-only --complete-trajectory
)
python -m efa.evaluate_episodes "${eval_args[@]}" \
  --root "$EFA_ROOT/data/derived/gqa_native_review_v2_complete" \
  --episode-file teacher_reviewed_episodes.jsonl \
  --output "$EFA_ROOT/runs/full_evidence_gqa_2b_v0"
exec python -m efa.evaluate_episodes "${eval_args[@]}" \
  --root "$EFA_ROOT/data/derived/clevr_count_program_v1" \
  --episode-file native_episodes.jsonl \
  --output "$EFA_ROOT/runs/full_evidence_clevr_2b_v0"
