#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=6000 {print $1,$2}' | sort -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then
  printf 'WAITING_FOR_GPU\n'
  exit 3
fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - <<'PY'
import json,math,time,torch
from pathlib import Path
from efa.qwen import QwenBackend,Sources
from efa.regions import Box
from efa.core import Claim
root=Path('/data_cinema/zqc/VBC')
data=root/'data/derived/zwz_seed_v0'
row=json.loads((data/'raw_tasks.jsonl').read_text().splitlines()[0])
b=QwenBackend(root/'models/Qwen2.5-VL-3B-Instruct',data/row['image'],True,max_pixels=50176)
claim=Claim('the line painted on the ground','color',row['answer'])
b.begin(claim)
source=Sources((Box(*row['bbox']),),(Box(*row['context_bbox']),),True)
torch.cuda.synchronize();start=time.monotonic()
value=b.generate(b.prompt(claim.query,list(source.target)),list(source.target),max_tokens=16)[0]
scores=b.score(claim,source)
torch.cuda.synchronize()
assert all(v is None or math.isfinite(v) for k,v in vars(scores).items() if k!='valid')
result=dict(task=row['id'],reference=row['answer'],generated=value,scores=vars(scores),
    source=b.describe_source(claim.query,source),seconds=time.monotonic()-start,
    peak_allocated_bytes=torch.cuda.max_memory_allocated(),calls=b.calls,
    region_source='native_annotation',model='base_3b_unadapted',calibrated=False,
    scope='backend_smoke_not_benchmark')
out=root/'runs/model_smoke.json'
if out.exists():
    raise RuntimeError('Preserve existing smoke result')
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result))
PY
