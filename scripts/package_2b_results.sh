#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
python - <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
root=Path('/data_cinema/zqc/VBC')
evaluation=root/'runs/heldout_2b_v0'
if not (evaluation/'summary.json').is_file():
    raise RuntimeError('Evaluation is not complete')
out=root/'data/derived/heldout_2b_v0'
out.mkdir(exist_ok=False)
files=list(evaluation.glob('*.json'))+list(evaluation.glob('*.jsonl'))
for path in files:
    shutil.copyfile(path,out/path.name)
for name,filename in [('region_sft_2b_v1','all_results.json'),
                      ('latent_reader_2b_v1','metrics.json'),
                      ('episode_sft_2b_v0','all_results.json'),
                      ('latent_episode_2b_v0','metrics.json')]:
    shutil.copyfile(root/'runs'/name/filename,out/(name+'-'+filename))
shutil.copyfile(root/'code/configs/policy_engineering.json',out/'policy.json')
shutil.copyfile(root/'data/derived/episode_sft_2b_v0/export_manifest.json',out/'episode_export_manifest.json')
summary=json.loads((out/'summary.json').read_text())
manifest=dict(kind='student_engineering_evaluation',count=summary['count'],
    images_included=False,calibrated=False,benchmark_dedup_status='pending',
    training_data='zwz_pilot_v1 + viscot_gqa_plus_v2 + chartqa_plus_v2',
    checkpoint=summary['checkpoint'],
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest))
PY
