"""Small Visual-CoT/GQA source slice with native programs and original VG images."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import urllib.error
import urllib.request
from PIL import Image
from efa.data import partition

REVISION='223d2d8c1146fda2bb918801b8276c587b78b61c'


def build(output,limit):
    root=Path(output)
    if root.exists():raise ValueError('Choose a new source version')
    url=f'https://hf-mirror.com/datasets/deepcs233/Visual-CoT/resolve/{REVISION}/cot_with_detailed_reasoning_steps/gqa_cot_train.jsonl'
    rows=[]
    with urllib.request.urlopen(url,timeout=60) as response:
        for index,line in enumerate(response):
            row=json.loads(line)
            if len(row.get('reasoning',[]))<4:continue
            rows.append((index,row))
            if len(rows)>=limit:break
    root.mkdir(parents=True);(root/'images').mkdir()

    def fetch(item):
        index,row=item;name=row['image']
        if Path(name).name!=name or not name.endswith('.jpg'):raise ValueError('Invalid VG filename')
        for folder in ('VG_100K','VG_100K_2'):
            image_url=f'https://cs.stanford.edu/people/rak248/{folder}/{name}'
            try:
                with urllib.request.urlopen(image_url,timeout=45) as response:raw=response.read()
                break
            except urllib.error.HTTPError as error:
                if error.code!=404:raise
        else:raise ValueError('Original image unavailable: '+name)
        image=Image.open(io.BytesIO(raw)).convert('RGB')
        digest=hashlib.sha256(image.tobytes()+str(image.size).encode()).hexdigest()
        path='images/'+digest+'.jpg';(root/path).write_bytes(raw)
        width,height=image.size
        boxes=[]
        for box in row['bboxs']:
            scaled=[round(v*(width/row['width'] if i%2==0 else height/row['height'])) for i,v in enumerate(box)]
            scaled=[max(0,min(v,width if i%2==0 else height)) for i,v in enumerate(scaled)]
            if scaled[0]<scaled[2] and scaled[1]<scaled[3]:boxes.append(scaled)
        return dict(id=f'viscot-gqa-{index}',source_repo='deepcs233/Visual-CoT',source_revision=REVISION,
            source_image_url=image_url,image=path,image_sha256=digest,width=width,height=height,
            question=row['question'],answer=row['answer'],source_program=row['reasoning'],
            native_candidate_boxes=boxes,source_annotation_frame=dict(width=row['width'],height=row['height'],boxes=row['bboxs']),
            source_split=row['split'],split=partition(digest),benchmark_dedup_status='pending',
            label_basis='GQA_program_and_VisualCoT_metadata_requires_step_evidence_review',
            annotation_status='source_qa_requires_episode_conversion',steps=None,runtime_audit=None)

    with ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(fetch,rows))
    (root/'raw_tasks.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    manifest=dict(kind='viscot_gqa_program_source',count=len(records),images=len({r['image'] for r in records}),
        source_revision=REVISION,construction_complete=True,complete_efa_episodes=0,
        benchmark_dedup_status='pending',image_edits=False,min_program_operations=4,
        license_note='Dataset card YAML says Apache-2.0; body says CC-BY-NC-4.0. Preserve both; research-only use. Original VG image rights remain with owners.')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=32)
    a=p.parse_args();build(a.output,a.limit)
