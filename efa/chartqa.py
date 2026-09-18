"""Bounded ChartQA training slice from original viewer images, grouped by pixels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
from PIL import Image
from efa.data import partition


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=16)
    a=p.parse_args();root=Path(a.output)
    if root.exists():raise ValueError('Choose a new version')
    url='https://datasets-server.huggingface.co/first-rows?dataset=HuggingFaceM4%2FChartQA&config=default&split=train'
    with urllib.request.urlopen(url,timeout=60) as response:data=json.load(response)
    selected=[r for r in data['rows'] if not r.get('truncated_cells') and re.search(
        r'\b(sum|difference|deduct|average|ratio|more than|greater than|exceed|below|total)\b',r['row']['query'],re.I)][:a.limit]
    root.mkdir(parents=True);(root/'images').mkdir()

    def fetch(entry):
        row=entry['row'];descriptor=row['image']
        with urllib.request.urlopen(descriptor['src'],timeout=60) as response:raw=response.read()
        decoded=Image.open(io.BytesIO(raw));image=decoded.convert('RGB')
        if image.size!=(descriptor['width'],descriptor['height']):raise ValueError('Viewer image dimensions differ')
        digest=hashlib.sha256(image.tobytes()+str(image.size).encode()).hexdigest()
        ext={'JPEG':'jpg','PNG':'png'}.get(decoded.format)
        if not ext:raise ValueError('Unexpected viewer image format')
        path='images/'+digest+'.'+ext;(root/path).write_bytes(raw)
        # The immutable source revision is part of the viewer image route.
        revision=descriptor['src'].split('/--/',1)[1].split('/',1)[0]
        return dict(id=f"chartqa-train-{entry['row_idx']}",source_repo='HuggingFaceM4/ChartQA',
            source_revision=revision,source_split='train',source_row=entry['row_idx'],
            image=path,image_sha256=digest,width=image.width,height=image.height,
            question=row['query'],answer=row['label'][0],reference_answers=row['label'],
            label_basis='ChartQA_training_reference_requires_step_evidence_review',
            human_or_machine=row['human_or_machine'],split=partition(digest),
            annotation_status='source_qa_requires_episode_conversion',benchmark_dedup_status='pending',steps=None,runtime_audit=None)
    with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,selected))
    (root/'raw_tasks.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    manifest=dict(kind='chartqa_train_arithmetic_slice',count=len(records),images=len({r['image'] for r in records}),
        source_revisions=sorted({r['source_revision'] for r in records}),construction_complete=True,
        source_split='train',complete_efa_episodes=0,benchmark_dedup_status='pending',image_edits=False)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))


if __name__=='__main__':main()
