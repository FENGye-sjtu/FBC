#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh

python - <<'PY'
import hashlib
import json
from pathlib import Path

root=Path('/data_cinema/zqc/VBC')
out=root/'data/derived/audit_backend_results_2b_v0'
if out.exists():
    raise FileExistsError('Preserve existing result version: '+str(out))

groups={
    'joint_compact': ('full_evidence_gqa_2b_v1_contract','full_evidence_clevr_2b_v1_contract'),
    'trajectory_compact': ('full_trajectory_reader_gqa_2b_v0_contract','full_trajectory_reader_clevr_2b_v0_contract'),
    'trajectory_native': ('native_crop_gqa_2b_v0_contract','native_crop_clevr_2b_v0_contract'),
}

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()

payload={}
comparison={}
for group,names in groups.items():
    comparison[group]={}
    for name in names:
        run=root/'runs'/name
        summary=read_json(run/'summary.json')
        rows=[json.loads(line) for line in (run/'results.jsonl').read_text().splitlines() if line.strip()]
        if summary.get('count')!=3 or len(rows)!=3:
            raise ValueError('Incomplete evaluation: '+name)
        if summary.get('calibrated') is not False or summary.get('engineering_only') is not True:
            raise ValueError('Evaluation provenance changed: '+name)
        settings=summary.get('evaluation_settings',{})
        if not settings.get('complete_trajectory') or settings.get('reacquire_on_binding_failure'):
            raise ValueError('Evaluation settings changed: '+name)
        checkpoint=summary.get('checkpoint',{})
        if group=='trajectory_native':
            if (summary.get('audit_backend')!='native_crop' or
                    settings.get('verifier_training')!='none' or
                    checkpoint.get('resampler_sha256') is not None or
                    checkpoint.get('representation')!='independently_encoded_native_crop_tokens'):
                raise ValueError('Primary native audit used a trained reader: '+name)
        for row in rows:
            for suffix,field in (('-trace.json','efa'),('-step-only.json','step_only')):
                trace=read_json(run/(row['id']+suffix))
                if trace.get('final')!=row[field]['final']:
                    raise ValueError('Trace/result mismatch: '+row['id'])
            if 'answer_contract' not in row['efa']['final']:
                raise ValueError('Final-answer contract missing: '+row['id'])
        comparison[group][name]={key:summary.get(key) for key in (
            'native_correct','audit_correct','compact_correct','direct_correct',
            'step_only_correct','efa_correct','verified','verified_correct',
            'verified_wrong','generated_steps','step_only_generated_steps')}
        for path in sorted(run.rglob('*')):
            if path.is_file() and path.suffix in {'.json','.jsonl'}:
                payload[path.relative_to(root).as_posix()]=path.read_bytes()

policy=root/'code/configs/policy_engineering.json'
payload['policy/policy_engineering.json']=policy.read_bytes()
payload['comparison.json']=(json.dumps(comparison,indent=2)+'\n').encode()
manifest={
    'kind':'matched_complete_trajectory_audit_backend_engineering_evaluation',
    'count':18,
    'groups':{key:list(value) for key,value in groups.items()},
    'primary_group':'trajectory_native',
    'primary_verifier_training':'none',
    'positive_negative_training_required':False,
    'complete_trajectory':True,
    'calibrated':False,
    'engineering_only':True,
    'benchmark_dedup_status':'pending',
    'construction_complete':True,
    'limitation':'Three GQA and three CLEVR development cases per checkpoint/backend; not a paper-scale efficacy result.',
    'files':{name:sha256_bytes(content) for name,content in sorted(payload.items())},
}
out.mkdir(parents=True)
for name,content in payload.items():
    target=out/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
for name,digest in manifest['files'].items():
    if sha256_bytes((out/name).read_bytes())!=digest:
        raise RuntimeError('Copied artifact hash mismatch: '+name)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(manifest),flush=True)
PY
