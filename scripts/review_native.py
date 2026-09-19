"""Review a bounded native-program slice with fresh source-only teacher calls."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.error
from efa.review import review


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=64);p.add_argument('--model',default='qwen3-vl-plus')
    p.add_argument('--cache-from',help='Reuse immutable responses from a preceding bounded pilot')
    a=p.parse_args();source=Path(a.source);out=Path(a.output)
    if a.limit<1:raise ValueError('Positive review limit required')
    rows=[json.loads(s) for s in (source/'native_episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    # Source order is already a seeded sample independent of model performance.
    rows=rows[:a.limit]
    identity=dict(model=a.model,ids=[r['id'] for r in rows],
        source_sha256=hashlib.sha256((source/'native_episodes.jsonl').read_bytes()).hexdigest(),
        review_sha256=hashlib.sha256(Path('efa/review.py').read_bytes()).hexdigest())
    if out.exists():
        if json.loads((out/'experiment.json').read_text())!=identity:raise ValueError('Existing experiment differs')
    else:
        out.mkdir(parents=True);(out/'teacher_cache').mkdir();(out/'images').mkdir()
        (out/'experiment.json').write_text(json.dumps(identity,indent=2))
        for row in rows:shutil.copyfile(source/row['image'],out/row['image'])
        (out/'raw_tasks.jsonl').write_text(''.join(json.dumps({k:v for k,v in r.items() if k!='steps'})+'\n' for r in rows))
        (out/'source_native_episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    if a.cache_from:
        for path in Path(a.cache_from).glob('*.json'):
            target=out/'teacher_cache'/path.name
            if not target.exists():shutil.copyfile(path,target)
    os.environ['EFA_TEACHER_BASE_URL']='https://dashscope.aliyuncs.com/compatible-mode/v1'
    os.environ['EFA_TEACHER_MODEL']=a.model
    results=out/'batch_results.jsonl'
    completed={r['id']:r for r in map(json.loads,results.read_text(encoding='utf-8').splitlines())} if results.exists() else {}
    pending=[r for r in rows if r['id'] not in completed or 'error' in completed[r['id']]]
    if pending:os.environ['EFA_TEACHER_API_KEY']=getpass.getpass('Teacher API key: ')
    def process(row):
        try:return dict(id=row['id'],reviewed=review(row,out,out/'teacher_cache'))
        except urllib.error.HTTPError as error:
            # Record the provider's error category, never request headers/body.
            try:
                detail=json.loads(error.read()).get('error',{})
                code=str(detail.get('code','unknown'))[:80]
            except (ValueError,AttributeError):code='unknown'
            return dict(id=row['id'],error=f'HTTP {error.code}',provider_code=code)
        except (ValueError,KeyError,TimeoutError,urllib.error.URLError) as error:
            return dict(id=row['id'],error=type(error).__name__+': '+str(error)[:200])
    with ThreadPoolExecutor(max_workers=4) as pool,results.open('a',encoding='utf-8') as stream:
        for result in pool.map(process,pending):
            stream.write(json.dumps(result)+'\n');stream.flush();completed[result['id']]=result
            print(result['id'],result.get('error',result.get('reviewed',{}).get('annotation_status')),flush=True)
    checked=[r['reviewed'] for r in completed.values() if 'reviewed' in r]
    (out/'teacher_reviewed_episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in checked))
    approved=[r for r in checked if r['annotation_status']=='teacher_reviewed_episode']
    # Imported caches can contain unrelated experiments. Count only requests
    # referenced by this slice, once each, including actual reused responses.
    requests={check['request_sha256']:check for row in checked for check in row['review']['checks']}
    usage=Counter()
    for check in requests.values():
        usage.update({k:v for k,v in (check.get('usage') or {}).items() if isinstance(v,int)})
    manifest=dict(kind='native_program_teacher_source_review',count=len(rows),reviewed=len(checked),
        approved=len(approved),approved_splits=dict(Counter(r['split'] for r in approved)),
        approved_steps=sum(len(r['steps']) for r in approved),errors=sum('error' in r for r in completed.values()),
        model=a.model,usage=dict(usage),usage_basis='unique_referenced_response_usage_including_reused_calls',
        referenced_requests=len(requests),
        benchmark_dedup_status='pending',runtime_verified_episodes=0,
        label_basis='native_programs_with_teacher_checks_not_ground_truth',construction_complete=len(checked)==len(rows))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
