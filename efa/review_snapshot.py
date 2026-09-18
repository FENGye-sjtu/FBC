"""Freeze completed source reviews while a larger offline batch continues."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,required=True);a=p.parse_args();source=Path(a.source);out=Path(a.output)
    if out.exists() or a.limit<1:raise ValueError('Use a fresh snapshot and positive limit')
    content=(source/'batch_results.jsonl').read_text(encoding='utf-8')
    # A trailing unfinished append is not a completed review.
    lines=content.splitlines() if content.endswith('\n') else content.splitlines()[:-1]
    latest={r['id']:r for r in map(json.loads,lines)}
    records=[r['reviewed'] for r in latest.values() if 'reviewed' in r][:a.limit]
    if len(records)!=a.limit:raise ValueError('Requested number of completed reviews is not yet available')
    out.mkdir(parents=True);(out/'images').mkdir()
    for row in records:shutil.copyfile(source/row['image'],out/row['image'])
    shutil.copyfile(source/'experiment.json',out/'experiment.json')
    (out/'teacher_reviewed_episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (out/'raw_tasks.jsonl').write_text(''.join(json.dumps({k:v for k,v in r.items() if k not in ['steps','review']})+'\n' for r in records))
    approved=[r for r in records if r['annotation_status']=='teacher_reviewed_episode']
    manifest=dict(kind='completed_review_snapshot',count=len(records),approved=len(approved),
        approved_splits=dict(Counter(r['split'] for r in approved)),approved_steps=sum(len(r['steps']) for r in approved),
        source_experiment_sha256=hashlib.sha256((source/'experiment.json').read_bytes()).hexdigest(),
        source_version=str(source),construction_complete=True,
        parent_batch_complete=json.loads((source/'manifest.json').read_text()).get('construction_complete',False) if (source/'manifest.json').exists() else False,
        runtime_verified_episodes=0,benchmark_dedup_status='pending',label_basis='native_programs_with_teacher_checks')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))


if __name__=='__main__':main()
