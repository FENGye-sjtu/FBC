#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
python - <<'PY'
import hashlib
import json
import math
from pathlib import Path
import yaml

root = Path('/data_cinema/zqc/VBC')
out = root / 'data/derived/mixed_program_results_v0'
if out.exists():
    raise FileExistsError('Preserve existing result version: ' + str(out))
sft = root / 'runs/mixed_program_sft_2b_v0'
reader = root / 'runs/latent_mixed_programs_2b_v0'

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

state = read_json(sft / 'trainer_state.json')
sft_metrics = read_json(sft / 'all_results.json')
precision = read_json(sft / 'precision_policy.json')
trainer_log = read_rows(sft / 'trainer_log.jsonl')
metrics = read_json(reader / 'metrics.json')
input_sources = read_json(reader / 'input_sources.json')
if state.get('global_step') != 2400 or state.get('max_steps') != 2400:
    raise ValueError('SFT must complete all 2400 optimizer updates')
if not math.isfinite(sft_metrics.get('train_loss', float('nan'))) or sft_metrics.get('train_runtime', 0) <= 0:
    raise ValueError('Missing finite completed SFT metrics')
if precision.get('mode') != 'bf16_vision_nf4_language':
    raise ValueError('Unexpected actual SFT precision policy')
if not trainer_log or trainer_log[-1].get('current_steps') != 2400:
    raise ValueError('Trainer log does not finish at update 2400')
if metrics.get('steps') != 1660 or metrics.get('examples') != 830 or metrics.get('episodes') != 344:
    raise ValueError('Reader must finish 1660 updates over 830 atoms from 344 episodes')
losses = metrics.get('losses', [])
if len(losses) != 1660 or not all(math.isfinite(loss) for loss in losses):
    raise ValueError('Reader loss record does not contain 1660 finite updates')
if input_sources.get('episodes') != 344 or metrics.get('input_sources') != input_sources.get('sources'):
    raise ValueError('Reader metrics and original input-source manifest disagree')

# Bind the archived summaries to the existing weights without publishing weights.
adapter_sha = sha256(sft / 'adapter_model.safetensors')
resampler_sha = sha256(reader / 'resampler.pt')
if metrics.get('encoder_config', {}).get('adapter_sha256') != adapter_sha:
    raise ValueError('Reader was aligned to a different SFT adapter')

# Use the policy recorded at launch. Never silently freeze a later edited policy.
launch_log = root / 'logs/mixed_programs_eval_v0.log'
launch_records = []
with launch_log.open(encoding='utf-8', errors='replace') as stream:
    for line_number, line in enumerate(stream, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get('scope') == 'old_native_and_teacher_data_mixed_sft_comparison' and 'policy_sha256' in value:
            launch_records.append(dict(line_number=line_number, record=value))
if not launch_records or len({row['record']['policy_sha256'] for row in launch_records}) != 1:
    raise ValueError('Original evaluation policy hash is missing or ambiguous in launch log')
policy_sha = launch_records[0]['record']['policy_sha256']
if any(row['record'].get('reader_steps') != 1660 or row['record'].get('calibrated') is not False for row in launch_records):
    raise ValueError('Launch metadata is not the uncalibrated 1660-update comparison')
policy_candidates = [
    root / 'runs/mixed_gqa_heldout_2b_v0/policy.json',
    root / 'runs/mixed_clevr_heldout_2b_v0/policy.json',
    root / 'code/configs/policy_engineering.json',
    root / 'data/derived/gqa_reader_objectives_eval_v1/policy.json',
]
policy_path = next((path for path in policy_candidates if path.is_file() and sha256(path) == policy_sha), None)
if policy_path is None:
    raise ValueError('No original policy snapshot matches the recorded evaluation hash')

cases = [
    ('mixed_gqa_heldout_2b_v0', 'teacher_reviewed_episodes.jsonl',
     ['gqa-native-9236', 'gqa-native-28788', 'gqa-native-94674']),
    ('mixed_clevr_heldout_2b_v0', 'native_episodes.jsonl',
     ['clevr-count-10563', 'clevr-count-263129', 'clevr-count-231696']),
]
summaries, source_files = {}, []
for name, tier, expected_ids in cases:
    run = root / 'runs' / name
    summary = read_json(run / 'summary.json')
    rows = read_rows(run / 'results.jsonl')
    if summary.get('count') != 3 or len(rows) != 3 or [row['id'] for row in rows] != expected_ids:
        raise ValueError('Incomplete or changed held-out cases: ' + name)
    if summary.get('split') != 'dev' or summary.get('episode_file') != tier:
        raise ValueError('Unexpected split or annotation tier: ' + name)
    if summary.get('calibrated') is not False or summary.get('engineering_only') is not True:
        raise ValueError('Evaluation must retain its uncalibrated engineering label')
    checkpoint = summary.get('checkpoint', {})
    if checkpoint.get('adapter_sha256') != adapter_sha or checkpoint.get('resampler_sha256') != resampler_sha:
        raise ValueError('Evaluation checkpoint hashes do not match completed mixed run')
    for row in rows:
        for suffix, field in [('-trace.json', 'efa'), ('-step-only.json', 'step_only')]:
            trace = read_json(run / (row['id'] + suffix))
            if trace.get('final') != row[field]['final']:
                raise ValueError('Saved final state differs from complete trace: ' + row['id'])
    for field, summary_field in [('direct', 'direct_correct'), ('step_only', 'step_only_correct'), ('efa', 'efa_correct')]:
        if sum(bool(row[field]['correct']) for row in rows) != summary.get(summary_field):
            raise ValueError('Results and aggregate summary disagree: ' + name)
    summaries[name] = summary
    # Retain all JSON/JSONL evaluation artifacts, including nested trace folders.
    source_files.extend(path for path in sorted(run.rglob('*')) if path.is_file() and path.suffix in {'.json', '.jsonl'})

for filename in ['all_results.json', 'precision_policy.json', 'trainer_log.jsonl', 'trainer_state.json']:
    source_files.append(sft / filename)
for filename in ['train_results.json', 'adapter_config.json']:
    if (sft / filename).is_file():
        source_files.append(sft / filename)
source_files += [reader / 'metrics.json', reader / 'input_sources.json']
export_manifest = root / 'data/derived/mixed_program_sft_2b_v0/export_manifest.json'
if export_manifest.is_file():
    source_files.append(export_manifest)

# All prerequisites were checked before creating the immutable output directory.
payload = {path.relative_to(root).as_posix(): path.read_bytes() for path in source_files}
payload['policy/policy_engineering.json'] = policy_path.read_bytes()
policy_provenance = dict(source_file=str(policy_path.relative_to(root)), policy_sha256=policy_sha,
    original_launch_log=str(launch_log.relative_to(root)), original_launch_log_sha256=sha256(launch_log),
    launch_records=launch_records, calibrated=False,
    note='Exact policy bytes match the pre-evaluation launch hash; no thresholds were retuned by this packager.')
payload['policy/provenance.json'] = (json.dumps(policy_provenance, indent=2) + '\n').encode('utf-8')
config_path = root / 'runs/mixed_program_sft_2b_v0.yaml'
config_bytes = config_path.read_bytes()
training_config = dict(config=yaml.safe_load(config_bytes), original_file=str(config_path.relative_to(root)),
    original_sha256=hashlib.sha256(config_bytes).hexdigest(), original_yaml=config_bytes.decode('utf-8'))
payload['runs/mixed_program_sft_2b_v0/training_config.json'] = (json.dumps(training_config, indent=2) + '\n').encode('utf-8')

manifest = dict(kind='mixed_program_sft_and_reader_engineering_evaluation', count=6,
    evaluation_counts={name: summary['count'] for name, summary in summaries.items()},
    evaluation_splits={name: summary['split'] for name, summary in summaries.items()},
    annotation_tiers={name: summary['episode_file'] for name, summary in summaries.items()},
    reference_basis={name: summary.get('reference_basis') for name, summary in summaries.items()},
    sft_updates=2400, reader_updates=1660, reader_visual_atoms=830, training_episodes=344,
    adapter_sha256=adapter_sha, resampler_sha256=resampler_sha, policy_sha256=policy_sha,
    precision='bf16_vision_nf4_language', calibrated=False, engineering_only=True,
    benchmark_dedup_status='pending', images_included=False, model_checkpoints_included=False,
    source_runs_modified=False, construction_complete=True,
    label_basis='old_native_program_and_teacher_reviewed_annotations_not_new_codex_adjudicated_data',
    limitation='Six development cases under original uncalibrated thresholds; not a formal benchmark or efficacy claim.',
    files={name: hashlib.sha256(content).hexdigest() for name, content in sorted(payload.items())})
out.mkdir(parents=True, exist_ok=False)
for name, content in payload.items():
    destination = out / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
for name, expected_sha in manifest['files'].items():
    if sha256(out / name) != expected_sha:
        raise RuntimeError('Copied artifact hash mismatch: ' + name)
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps(manifest))
PY
