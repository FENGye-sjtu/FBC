"""Bounded teacher drafts to be individually inspected by Codex, never auto-approved."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import getpass
import json
import os
from pathlib import Path
import urllib.error
from efa.annotate import annotate_task


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--limit',type=int,default=32);p.add_argument('--model',default='qwen3-vl-plus')
    a=p.parse_args();root=Path(a.root)
    if a.limit<1:raise ValueError('Positive limit required')
    rows=[json.loads(s) for s in (root/'raw_tasks.jsonl').read_text(encoding='utf-8').splitlines()][:a.limit]
    out=root/'teacher_draft_results.jsonl';cache=root/'teacher_cache';cache.mkdir(exist_ok=True)
    completed={r['id']:r for r in map(json.loads,out.read_text(encoding='utf-8').splitlines())} if out.exists() else {}
    pending=[r for r in rows if r['id'] not in completed]
    if not pending:print('All bounded requests recorded');return
    os.environ.update(EFA_TEACHER_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1',EFA_TEACHER_MODEL=a.model)
    os.environ['EFA_TEACHER_API_KEY']=getpass.getpass('Teacher API key: ')
    def draft(row):
        try:return dict(id=row['id'],draft=annotate_task(row,root,cache))
        except urllib.error.HTTPError as error:
            return dict(id=row['id'],error='HTTP '+str(error.code))
        except (ValueError,KeyError,TypeError,TimeoutError,urllib.error.URLError) as error:
            return dict(id=row['id'],error=type(error).__name__+': '+str(error)[:160])
    with ThreadPoolExecutor(max_workers=4) as pool,out.open('a',encoding='utf-8') as stream:
        for result in pool.map(draft,pending):
            stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
            print(result['id'],result.get('error','draft_requires_individual_codex_review'),flush=True)


if __name__=='__main__':main()
