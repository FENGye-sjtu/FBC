"""Reuse original DeepEyesV2 tasks without deploying its search/code tools."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
from PIL import Image
from efa.data import partition

REVISION='cee88d9e9c494729f719ccca5de5aeb940501a35'


def build(source,output,limit):
    source=Path(source);output=Path(output)
    if output.exists():raise ValueError('Choose a new output version')
    rows=json.loads((source/'json/sft_part_0.json').read_text())
    output.mkdir(parents=True);(output/'images').mkdir()
    records=[];rejected={}
    with zipfile.ZipFile(source/'images.zip') as archive:
        names=set(archive.namelist())
        for index,row in enumerate(rows):
            messages=row['messages']
            # Tool results and historical system instructions are data, not
            # commands for this converter or the deployed student.
            if any(m['role']=='assistant' and '<tool_call>' in m['content'] for m in messages):
                rejected['external_tool_call']=rejected.get('external_tool_call',0)+1;continue
            users=[m['content'] for m in messages if m['role']=='user']
            answers=[m['content'] for m in messages if m['role']=='assistant']
            if not users or not answers or users[0].count('<image>')!=1 or not row.get('images'):
                continue
            matches=re.findall(r'<answer>(.*?)</answer>',answers[-1],re.S)
            if not matches:continue
            question=users[0].split('You must put your answer inside')[0].replace('<image>','').strip()
            answer=matches[-1].strip()
            if not question or not answer:continue
            name=row['images'][0]
            if name not in names:
                name=name.removeprefix('images/')
            if name not in names:continue
            raw=archive.read(name)
            decoded=Image.open(io.BytesIO(raw));image=decoded.convert('RGB')
            digest=hashlib.sha256(image.tobytes()+str(image.size).encode()).hexdigest()
            ext={'JPEG':'jpg','PNG':'png','WEBP':'webp'}.get(decoded.format)
            path='images/'+digest+'.'+(ext or 'png')
            if ext:(output/path).write_bytes(raw)
            else:image.save(output/path)
            records.append(dict(id=f'deepeyesv2-p0-{index}',source_repo='honglyhly/DeepEyesV2_SFT',
                source_revision=REVISION,source_image=name,image=path,image_sha256=digest,
                width=image.width,height=image.height,question=question,answer=answer,
                source_trace=answers,split=partition(digest),annotation_status='source_qa_requires_episode_conversion',
                benchmark_dedup_status='pending',label_basis='upstream_teacher_answer_not_ground_truth',
                steps=None,runtime_audit=None))
            if len(records)>=limit:break
    (output/'raw_tasks.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    manifest=dict(kind='deepeyesv2_source_qa',count=len(records),images=len({r['image_sha256'] for r in records}),
        source_revision=REVISION,complete_efa_episodes=0,construction_complete=True,
        benchmark_dedup_status='pending',rejected=rejected,image_edits=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=64)
    a=p.parse_args();build(a.source,a.output,a.limit)
