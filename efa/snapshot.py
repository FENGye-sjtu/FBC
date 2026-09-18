"""Freeze a consistent prefix of a live region build for training/publication."""
import argparse
import json
from pathlib import Path
import shutil
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=128)
    a=p.parse_args();source=Path(a.source);output=Path(a.output)
    if output.exists() or a.limit<1:
        raise ValueError('Use a new snapshot directory and positive limit')
    for attempt in range(5):
        try:
            manifest=json.loads((source/'manifest.json').read_text())
            rows=[json.loads(s) for s in (source/'raw_tasks.jsonl').read_text().splitlines()]
            if len(rows)!=manifest['count']:
                raise ValueError('Live checkpoint changed during read')
            break
        except ValueError:
            if attempt==4: raise
            time.sleep(.2)
    if len(rows)<a.limit:
        raise ValueError(f'Only {len(rows)} records available; need {a.limit}')
    rows=rows[:a.limit];output.mkdir(parents=True)
    for name in {r['image'] for r in rows}:
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Image must be source-relative')
        dest=output/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/rel,dest)
    (output/'raw_tasks.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    manifest.update(kind='region_qa_snapshot',count=len(rows),images=len({r['image_sha256'] for r in rows}),
                    construction_complete=True,snapshot_of=str(source),requested_count=a.limit)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'README.md').write_text(
        '# EFA region pilot snapshot\n\n'+str(len(rows))+' region QA records from inclusionAI/ZwZ-RL-VQA. '
        'Complete EFA episodes: 0. Source revision and image hashes are in the records. '
        'Labels retain upstream teacher-consensus status; benchmark/near-duplicate clearance is pending. '
        'Original images are not synthetically edited. This is an engineering snapshot, not a benchmark result. '
        'Upstream dataset release declares Apache-2.0; original image rights remain with their owners.\n')
    print(json.dumps(manifest))


if __name__=='__main__':
    main()
