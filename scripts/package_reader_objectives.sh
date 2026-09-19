#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
python - <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
root=Path('/data_cinema/zqc/VBC')
source=root/'runs/gqa_reader_objectives_eval_v1'
summary=json.loads((source/'summary.json').read_text())
training=root/'runs/reader_objectives_2b_v1'
metrics=json.loads((training/'metrics.json').read_text())
out=root/'data/derived/gqa_reader_objectives_eval_v1'
out.mkdir(exist_ok=False)
for path in list(source.glob('*.json'))+list(source.glob('*.jsonl')):
    shutil.copyfile(path,out/path.name)
for filename in ['metrics.json','losses.jsonl']:
    shutil.copyfile(training/filename,out/('training-'+filename))
shutil.copyfile(root/'runs/gqa_heldout_2b_v2/summary.json',out/'baseline-summary.json')
shutil.copyfile(root/'code/configs/policy_engineering.json',out/'policy.json')
manifest=dict(kind='reader_objective_bundle_engineering_evaluation',count=summary['count'],
    source_version='gqa_native_review_v2_complete',source_split=summary['split'],
    training_version='gqa_reader_objectives_v1',training_steps=metrics['steps'],
    comparison='same_adapter_cases_and_policy_changed_reader_objective_bundle',
    calibrated=False,benchmark_dedup_status='pending',images_included=False,
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest))
PY
