#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
python - <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import yaml
root=Path('/data_cinema/zqc/VBC');source=root/'runs/gqa_heldout_2b_v0'
summary=json.loads((source/'summary.json').read_text())
out=root/'data/derived/gqa_heldout_2b_v0';out.mkdir(exist_ok=False)
for path in list(source.glob('*.json'))+list(source.glob('*.jsonl')):
    shutil.copyfile(path,out/path.name)
for name,filename in [('gqa_sft_2b_v0','all_results.json'),('latent_gqa_2b_v0','metrics.json')]:
    shutil.copyfile(root/'runs'/name/filename,out/(name+'-'+filename))
shutil.copyfile(root/'code/configs/policy_engineering.json',out/'policy.json')
(out/'training_config.json').write_text(json.dumps(yaml.safe_load((root/'runs/gqa_sft_2b_v0.yaml').read_text()),indent=2))
manifest=dict(kind='heldout_source_reviewed_episode_evaluation',count=summary['count'],
    source_version='gqa_native_review_v1',source_split='calibration',
    calibrated=False,benchmark_dedup_status='pending',images_included=False,
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
PY
