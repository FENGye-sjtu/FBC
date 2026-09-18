#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=6000 {print $1,$2}' | sort -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then exit 3; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - <<'PY'
import json
from pathlib import Path
from efa.latent_backend import LatentBackend
from efa.core import Policy
from efa.infer import run
root=Path('/data_cinema/zqc/VBC')
data=root/'data/derived/zwz_seed_v0'
task=json.loads((data/'raw_tasks.jsonl').read_text().splitlines()[0])
backend=LatentBackend(root/'models/Qwen2.5-VL-3B-Instruct',data/task['image'],True,max_pixels=50176,
    adapter_path=root/'runs/step_sft_smoke_v0',resampler_path=root/'runs/latent_reader_smoke_v1/resampler.pt')
config=json.loads((root/'code/configs/policy_engineering.json').read_text())
result=run(backend,task['question'],{k:Policy(**v) for k,v in config['strata'].items()})
result['task_id']=task['id']
result['reference_answer']=task['answer']
result['scope']='training_image_episode_engineering_smoke_not_benchmark'
out=root/'runs/episode_smoke_v1.json'
if out.exists():
    raise RuntimeError('Preserve previous episode')
out.write_text(json.dumps(result,indent=2))
print(json.dumps(dict(final=result['final'],steps=len(result['trace']),budget=result['budget'],calls=len(result['cost']))))
PY
