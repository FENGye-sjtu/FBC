#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"

python - <<'PY'
import json
from pathlib import Path

root=Path('/data_cinema/zqc/VBC')
adapter=root/'runs/codex_full_trajectory_2b_v0'
state=json.loads((adapter/'trainer_state.json').read_text())
if state.get('global_step')!=114 or abs(state.get('epoch',0)-3.0)>1e-6:
    raise ValueError('Completed full-trajectory SFT adapter is required')
for name in ('native_crop_gqa_2b_v0_contract','native_crop_clevr_2b_v0_contract'):
    if (root/'runs'/name).exists():
        raise ValueError('Preserve existing run: '+name)
print(json.dumps({
    'adapter':str(adapter),
    'audit_backend':'native_crop',
    'verifier_training':'none',
    'positive_negative_training':False,
    'complete_trajectory':True,
}),flush=True)
PY

gpu=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
  | awk -F, '{gsub(/ /,""); if ($2>=7000) print $1,$2,$3}' \
  | sort -k3n -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then echo 'WAIT_FOR_7000_MIB_FREE'; exit 3; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -i "$gpu" --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader

eval_args=(
  --model "$EFA_ROOT/models/Qwen2-VL-2B-Instruct"
  --adapter "$EFA_ROOT/runs/codex_full_trajectory_2b_v0"
  --audit-backend native_crop
  --policy "$EFA_ROOT/code/configs/policy_engineering.json"
  --split dev --limit 3 --engineering-only --complete-trajectory
)
python -m efa.evaluate_episodes "${eval_args[@]}" \
  --root "$EFA_ROOT/data/derived/gqa_native_review_v2_complete" \
  --episode-file teacher_reviewed_episodes.jsonl \
  --output "$EFA_ROOT/runs/native_crop_gqa_2b_v0_contract"
exec python -m efa.evaluate_episodes "${eval_args[@]}" \
  --root "$EFA_ROOT/data/derived/clevr_count_program_v1" \
  --episode-file native_episodes.jsonl \
  --output "$EFA_ROOT/runs/native_crop_clevr_2b_v0_contract"
