#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh

python - <<'PY'
import hashlib
import json
import math
from pathlib import Path
import shutil

root = Path('/data_cinema/zqc/VBC')
out = root / 'data/derived/full_evidence_results_2b_v0'
if out.exists():
    raise FileExistsError('Preserve existing result version: ' + str(out))

sft = root / 'runs/codex_full_trajectory_2b_v0'
latent = root / 'runs/latent_full_trajectory_2b_v0'
joint = root / 'runs/full_evidence_2b_v0'
evaluations = {
    'full_evidence_gqa_2b_v0': ('teacher_reviewed_episodes.jsonl', [
        'gqa-native-9236', 'gqa-native-28788', 'gqa-native-94674']),
    'full_evidence_clevr_2b_v0': ('native_episodes.jsonl', [
        'clevr-count-10563', 'clevr-count-263129', 'clevr-count-231696']),
}

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

sft_state = read_json(sft / 'trainer_state.json')
sft_metrics = read_json(sft / 'all_results.json')
sft_precision = read_json(sft / 'precision_policy.json')
sft_manifest_path = root / 'data/derived/codex_full_trajectory_2b_v0/export_manifest.json'
sft_manifest = read_json(sft_manifest_path)
latent_metrics = read_json(latent / 'metrics.json')
joint_metrics = read_json(joint / 'metrics.json')
joint_sources = read_json(joint / 'input_sources.json')

if sft_state.get('global_step') != 114 or sft_state.get('epoch') != 3.0:
    raise ValueError('Complete-trajectory SFT did not finish exactly 114 updates')
if sft_manifest.get('sft_examples') != 75 or sft_manifest.get('assistant_turns') != 127:
    raise ValueError('Complete-trajectory SFT composition changed')
if not math.isfinite(sft_metrics.get('train_loss', float('nan'))) or sft_metrics.get('train_runtime', 0) <= 0:
    raise ValueError('Missing finite completed SFT metrics')
if sft_precision.get('mode') != 'bf16_vision_nf4_language':
    raise ValueError('Unexpected SFT precision policy')
if (latent_metrics.get('steps'), latent_metrics.get('examples'), latent_metrics.get('episodes')) != (424, 106, 75):
    raise ValueError('Independent reader alignment is incomplete or changed')
if len(latent_metrics.get('losses', [])) != 424 or not all(math.isfinite(x) for x in latent_metrics['losses']):
    raise ValueError('Reader losses are incomplete or non-finite')
if (joint_metrics.get('steps'), joint_metrics.get('examples'), joint_metrics.get('correction_pairs'),
        joint_metrics.get('positive_replay')) != (244, 122, 16, 106):
    raise ValueError('Joint correction/replay training is incomplete or changed')
if joint_metrics.get('actual_student_repairs') != 0 or joint_metrics.get('calibrated') is not False:
    raise ValueError('Joint run provenance changed')
if joint_metrics.get('input_sources') != joint_sources:
    raise ValueError('Joint metrics and input-source manifest disagree')

sft_adapter_sha = sha256(sft / 'adapter_model.safetensors')
joint_adapter_sha = sha256(joint / 'adapter/adapter_model.safetensors')
latent_resampler_sha = sha256(latent / 'resampler.pt')
joint_resampler_sha = sha256(joint / 'resampler.pt')
if sft_manifest.get('parent_adapter_sha256') == sft_adapter_sha:
    raise ValueError('SFT output unexpectedly equals the parent adapter hash')
if latent_metrics.get('encoder_config', {}).get('adapter_sha256') != sft_adapter_sha:
    raise ValueError('Reader was not aligned to the completed trajectory adapter')
if joint_metrics.get('encoder_config', {}).get('adapter_sha256') != joint_adapter_sha:
    raise ValueError('Joint metrics do not identify the saved joint adapter')

summaries = {}
source_files = []
for name, (tier, expected_ids) in evaluations.items():
    run = root / 'runs' / name
    summary = read_json(run / 'summary.json')
    rows = read_rows(run / 'results.jsonl')
    if summary.get('count') != 3 or len(rows) != 3 or [row['id'] for row in rows] != expected_ids:
        raise ValueError('Incomplete or changed held-out cases: ' + name)
    if summary.get('split') != 'dev' or summary.get('episode_file') != tier:
        raise ValueError('Unexpected split or annotation tier: ' + name)
    if summary.get('evaluation_settings') != {
            'reacquire_on_binding_failure': False, 'complete_trajectory': True}:
        raise ValueError('Evaluation did not preserve complete-trajectory history: ' + name)
    if summary.get('calibrated') is not False or summary.get('engineering_only') is not True:
        raise ValueError('Evaluation must remain uncalibrated engineering-only')
    checkpoint = summary.get('checkpoint', {})
    if checkpoint.get('adapter_sha256') != joint_adapter_sha or checkpoint.get('resampler_sha256') != joint_resampler_sha:
        raise ValueError('Evaluation checkpoint hashes disagree: ' + name)
    for row in rows:
        for suffix, field in [('-trace.json', 'efa'), ('-step-only.json', 'step_only')]:
            trace = read_json(run / (row['id'] + suffix))
            if trace.get('final') != row[field]['final']:
                raise ValueError('Saved final differs from result row: ' + row['id'])
    for field, summary_field in [('direct', 'direct_correct'), ('step_only', 'step_only_correct'), ('efa', 'efa_correct')]:
        if sum(bool(row[field]['correct']) for row in rows) != summary.get(summary_field):
            raise ValueError('Aggregate summary disagrees with rows: ' + name)
    summaries[name] = summary
    source_files.extend(path for path in sorted(run.rglob('*')) if path.is_file() and path.suffix in {'.json', '.jsonl'})

baseline_root = root / 'data/derived/mixed_program_results_v0/runs'
baseline_names = {
    'full_evidence_gqa_2b_v0': 'mixed_gqa_heldout_2b_v0',
    'full_evidence_clevr_2b_v0': 'mixed_clevr_heldout_2b_v0',
}
comparison = {}
for current_name, baseline_name in baseline_names.items():
    baseline = read_json(baseline_root / baseline_name / 'summary.json')
    current = summaries[current_name]
    if baseline.get('episode_file_sha256') != current.get('episode_file_sha256') or baseline.get('count') != 3:
        raise ValueError('Baseline is not the same held-out slice: ' + baseline_name)
    fields = ['native_correct', 'compact_correct', 'direct_correct', 'step_only_correct',
              'efa_correct', 'verified', 'verified_correct', 'verified_wrong']
    comparison[current_name] = {
        field: {'baseline': baseline.get(field, 0), 'current': current.get(field, 0)}
        for field in fields
    }

for path in [
    sft / 'adapter_model.safetensors', sft / 'adapter_config.json',
    sft / 'all_results.json', sft / 'precision_policy.json', sft / 'trainer_state.json',
    latent / 'resampler.pt', latent / 'metrics.json',
    joint / 'adapter/adapter_model.safetensors', joint / 'adapter/adapter_config.json',
    joint / 'resampler.pt', joint / 'metrics.json', joint / 'input_sources.json', joint / 'losses.jsonl',
    sft_manifest_path,
]:
    if not path.is_file():
        raise FileNotFoundError(path)
    source_files.append(path)
for optional in [sft / 'train_results.json', sft / 'trainer_log.jsonl']:
    if optional.is_file():
        source_files.append(optional)

policy = root / 'code/configs/policy_engineering.json'
policy_sha = sha256(policy)
payload = {path.relative_to(root).as_posix(): path.read_bytes() for path in source_files}
payload['policy/policy_engineering.json'] = policy.read_bytes()
payload['comparison.json'] = (json.dumps(comparison, indent=2) + '\n').encode('utf-8')

manifest = {
    'kind': 'complete_trajectory_sft_joint_evidence_engineering_evaluation',
    'count': 6,
    'training_episodes': 75,
    'assistant_turns': 127,
    'visual_atoms': 106,
    'sft_updates': 114,
    'reader_updates': 424,
    'joint_updates': 244,
    'correction_pairs': 16,
    'positive_replay': 106,
    'actual_student_repairs': 0,
    'evaluation_counts': {name: value['count'] for name, value in summaries.items()},
    'sft_adapter_sha256': sft_adapter_sha,
    'joint_adapter_sha256': joint_adapter_sha,
    'latent_resampler_sha256': latent_resampler_sha,
    'joint_resampler_sha256': joint_resampler_sha,
    'policy_sha256': policy_sha,
    'precision': 'bf16_vision_nf4_language',
    'complete_trajectory': True,
    'calibrated': False,
    'engineering_only': True,
    'benchmark_dedup_status': 'pending',
    'images_included': False,
    'model_checkpoints_included': True,
    'base_model_included': False,
    'construction_complete': True,
    'limitation': 'Six fixed development cases under uncalibrated thresholds; this run regresses against the archived mixed baseline and is not an efficacy claim.',
    'files': {name: hashlib.sha256(content).hexdigest() for name, content in sorted(payload.items())},
}

out.mkdir(parents=True, exist_ok=False)
for name, content in payload.items():
    destination = out / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
for name, expected in manifest['files'].items():
    if sha256(out / name) != expected:
        raise RuntimeError('Copied artifact hash mismatch: ' + name)
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps(manifest))
PY
