#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"

python - <<'PY'
import json
from pathlib import Path

root = Path('/data_cinema/zqc/VBC')
adapter = root / 'runs/codex_full_trajectory_2b_v0'
reader = root / 'runs/latent_full_trajectory_2b_v0'
state = json.loads((adapter / 'trainer_state.json').read_text())
metrics = json.loads((reader / 'metrics.json').read_text())
if state.get('global_step') != 114 or metrics.get('steps') != 424:
    raise ValueError('Completed trajectory adapter and aligned reader required')
for name in ('full_trajectory_reader_gqa_2b_v0_contract', 'full_trajectory_reader_clevr_2b_v0_contract'):
    if (root / 'runs' / name).exists():
        raise ValueError('Preserve existing evaluation: ' + name)
print(json.dumps({'adapter': str(adapter), 'reader': str(reader),
                  'controller': 'question_level_final_premise_audit'}))
PY

gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=7000 {print $1,$2}' | sort -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then echo 'WAIT_FOR_7000_MIB_FREE'; exit 3; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -i "$gpu" --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader

eval_args=(
  --model "$EFA_ROOT/models/Qwen2-VL-2B-Instruct"
  --adapter "$EFA_ROOT/runs/codex_full_trajectory_2b_v0"
  --resampler "$EFA_ROOT/runs/latent_full_trajectory_2b_v0/resampler.pt"
  --policy "$EFA_ROOT/code/configs/policy_engineering.json"
  --split dev --limit 3 --engineering-only --complete-trajectory
)
python -m efa.evaluate_episodes "${eval_args[@]}" \
  --root "$EFA_ROOT/data/derived/gqa_native_review_v2_complete" \
  --episode-file teacher_reviewed_episodes.jsonl \
  --output "$EFA_ROOT/runs/full_trajectory_reader_gqa_2b_v0_contract"
exec python -m efa.evaluate_episodes "${eval_args[@]}" \
  --root "$EFA_ROOT/data/derived/clevr_count_program_v1" \
  --episode-file native_episodes.jsonl \
  --output "$EFA_ROOT/runs/full_trajectory_reader_clevr_2b_v0_contract"
