#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh

python - <<'PY'
from collections import deque
import json
from pathlib import Path
import subprocess

root = Path('/data_cinema/zqc/VBC')
needles = [
    'full_evidence_2b_v0',
    'latent_full_trajectory_2b_v0',
    'full_evidence_gqa_2b_v0',
    'full_evidence_clevr_2b_v0',
    'full_evidence_gqa_2b_v1_contract',
    'full_evidence_clevr_2b_v1_contract',
    'full_trajectory_reader_gqa_2b_v0_contract',
    'full_trajectory_reader_clevr_2b_v0_contract',
    'native_crop_gqa_2b_v0_contract',
    'native_crop_clevr_2b_v0_contract',
]

print('GPU')
print(subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu',
    '--format=csv,noheader'], text=True).strip())

print('PROCESSES')
matched = False
for line in subprocess.check_output(['ps', '-eo', 'pid,ppid,etime,args'], text=True).splitlines():
    if any(name in line for name in needles):
        matched = True
        print(line)
if not matched:
    print('none')

print('ARTIFACTS')
for relative in [
    'runs/latent_full_trajectory_2b_v0/metrics.json',
    'runs/full_evidence_2b_v0/metrics.json',
    'runs/full_evidence_gqa_2b_v0/summary.json',
    'runs/full_evidence_clevr_2b_v0/summary.json',
    'runs/full_evidence_gqa_2b_v1_contract/summary.json',
    'runs/full_evidence_clevr_2b_v1_contract/summary.json',
    'runs/full_trajectory_reader_gqa_2b_v0_contract/summary.json',
    'runs/full_trajectory_reader_clevr_2b_v0_contract/summary.json',
    'runs/native_crop_gqa_2b_v0_contract/summary.json',
    'runs/native_crop_clevr_2b_v0_contract/summary.json',
]:
    path = root / relative
    if not path.exists():
        print(relative, 'MISSING')
        continue
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        print(relative, 'INCOMPLETE_JSON')
        continue
    if relative.endswith('metrics.json'):
        losses = value.get('losses', [])
        summary = {
            key: value[key] for key in (
                'steps', 'examples', 'seconds', 'peak_allocated_bytes',
            ) if key in value
        }
        if losses:
            summary.update({
                'loss_count': len(losses),
                'loss_first': losses[0],
                'loss_last': losses[-1],
                'loss_mean': sum(losses) / len(losses),
            })
    else:
        summary = {key: value.get(key) for key in (
            'count', 'visual_atoms', 'native_correct', 'compact_correct',
            'direct_correct', 'step_only_correct', 'efa_correct', 'verified',
            'verified_correct', 'verified_wrong', 'generated_steps',
            'step_only_generated_steps', 'audit_backend', 'audit_correct',
            'evaluation_settings',
        )}
        checkpoint = value.get('checkpoint', {})
        summary['adapter_sha256'] = checkpoint.get('adapter_sha256')
        summary['resampler_sha256'] = checkpoint.get('resampler_sha256')
    print(relative, json.dumps(summary, ensure_ascii=True))

print('LOG_TAIL')
for relative in ('logs/full_evidence_2b_v0.log', 'logs/full_evidence_contract_2b_v1.log',
                 'logs/full_trajectory_reader_contract_2b_v0.log',
                 'logs/native_crop_contract_2b_v0.log'):
    path = root / relative
    print(relative)
    if path.exists():
        with path.open(errors='replace') as stream:
            for line in deque(stream, maxlen=18):
                print(line.rstrip()[:2000])
    else:
        print('missing')
PY
