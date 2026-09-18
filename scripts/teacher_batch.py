"""Bounded, resumable teacher experiment with per-stage accounting and no stored key."""
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
from efa.annotate import annotate_task
from efa.review import review, usable_sources


def run():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',required=True);p.add_argument('--limit',type=int,default=8)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--cache-from',action='append',default=[])
    p.add_argument('--retry-errors',action='store_true')
    a=p.parse_args()
    if a.limit<1 or not 1<=a.workers<=4:raise ValueError('Invalid bounded batch size')
    os.environ['EFA_TEACHER_BASE_URL']='https://dashscope.aliyuncs.com/compatible-mode/v1'
    os.environ['EFA_TEACHER_MODEL']=a.model
    os.environ['EFA_TEACHER_API_KEY']=getpass.getpass('Teacher API key: ')
    source=Path(a.source).resolve();root=Path(a.output).resolve()
    rows=[json.loads(s) for s in (source/'raw_tasks.jsonl').read_text(encoding='utf-8').splitlines()][:a.limit]
    identity=dict(model=a.model,source=str(source),ids=[r['id'] for r in rows],
                  annotate_sha256=hashlib.sha256(Path('efa/annotate.py').read_bytes()).hexdigest(),
                  review_sha256=hashlib.sha256(Path('efa/review.py').read_bytes()).hexdigest())
    if root.exists():
        if json.loads((root/'experiment.json').read_text())!=identity:
            raise ValueError('Existing experiment differs; choose a new version')
    else:
        root.mkdir(parents=True);(root/'teacher_cache').mkdir()
        for row in rows:
            path=Path(row['image'])
            if path.is_absolute() or '..' in path.parts:raise ValueError('Invalid image path')
            dest=root/path;dest.parent.mkdir(exist_ok=True,parents=True)
            shutil.copy2(source/path,dest)
        (root/'raw_tasks.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        (root/'experiment.json').write_text(json.dumps(identity,indent=2))
    for directory in a.cache_from:
        for cached in Path(directory).glob('*.json'):
            dest=root/'teacher_cache'/cached.name
            if not dest.exists():shutil.copy2(cached,dest)
    resultfile=root/'batch_results.jsonl'
    existing=[json.loads(s) for s in resultfile.read_text(encoding='utf-8').splitlines()] if resultfile.exists() else []
    existing=list({r['id']:r for r in existing}.values())
    done={r['id'] for r in existing if not (a.retry_errors and 'error' in r)}

    def process(row):
        stage='annotation'
        episode=None
        try:
            episode=annotate_task(row,root,root/'teacher_cache')
            stage='review'
            checked=review(episode,root,root/'teacher_cache')
            return dict(id=row['id'],stage='completed',episode=episode,reviewed=checked)
        except urllib.error.HTTPError as error:
            result=dict(id=row['id'],stage=stage,error=f'HTTP {error.code}')
        except (ValueError,KeyError,TypeError,TimeoutError,urllib.error.URLError) as error:
            result=dict(id=row['id'],stage=stage,error=type(error).__name__+': '+str(error)[:240])
        if episode is not None:result['episode']=episode
        return result

    with ThreadPoolExecutor(max_workers=a.workers) as pool,resultfile.open('a',encoding='utf-8') as out:
        for result in pool.map(process,[r for r in rows if r['id'] not in done]):
            out.write(json.dumps(result,ensure_ascii=False)+'\n');out.flush();existing.append(result)
            print(result['id'],result.get('error',result.get('reviewed',{}).get('annotation_status')),flush=True)
    existing=list({r['id']:r for r in existing}.values())
    episodes=[r['episode'] for r in existing if 'episode' in r]
    checked=[r['reviewed'] for r in existing if 'reviewed' in r]
    for name,values in [('pseudo_episodes',episodes),('teacher_reviewed_episodes',checked)]:
        (root/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values),encoding='utf-8')
    approved=[r for r in checked if r['annotation_status']=='teacher_reviewed_episode' and usable_sources(r)]
    usage=Counter()
    # Cached usage includes imported responses; it is not new billed usage.
    for f in (root/'teacher_cache').glob('*.json'):
        cached=json.loads(f.read_text(encoding='utf-8'))
        if cached['model']==a.model:
            usage.update({k:v for k,v in (cached.get('usage') or {}).items() if isinstance(v,int)})
    manifest=dict(kind='teacher_experiment',count=len(rows),images=len({r['image'] for r in rows}),
        construction_complete=True,source=str(source),model=a.model,teacher_generated_episodes=len(episodes),
        teacher_reviewed_pseudo_episodes=len(approved),approved_multistep_episodes=sum(len(r['steps'])>1 for r in approved),
        approved_splits=dict(Counter(r['split'] for r in approved)),errors=dict(Counter(r['stage'] for r in existing if 'error' in r)),
        usage=dict(usage),usage_basis='cached_response_usage_including_reused_calls',
        complete_efa_episodes=0,runtime_verified_episodes=0,
        label_basis='teacher_pseudo_labels_not_ground_truth',benchmark_dedup_status='pending')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':run()
