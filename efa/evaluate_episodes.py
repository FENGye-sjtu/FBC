"""Evaluate source-reviewed held-out steps and fully generated EFA trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
from efa.core import Policy
from efa.evaluate import canonical
from efa.infer import run
from efa.latent_backend import LatentBackend
from efa.qwen import QwenBackend,Sources
from efa.regions import Box


def main():
    p=argparse.ArgumentParser()
    for key in ['model','adapter','resampler','root','output','policy']:p.add_argument('--'+key,required=True)
    p.add_argument('--split',choices=['dev','calibration'],default='dev')
    p.add_argument('--limit',type=int,default=2);p.add_argument('--engineering-only',action='store_true')
    p.add_argument('--quantize-vision',action='store_true',help='Historical NF4-vision checkpoint replay')
    a=p.parse_args();root=Path(a.root);out=Path(a.output)
    if out.exists() or a.limit<1:raise ValueError('Fresh output and positive case limit required')
    all_rows=[json.loads(s) for s in (root/'teacher_reviewed_episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    train_images={r['image_sha256'] for r in all_rows if r['split']=='train'}
    rows=sorted((r for r in all_rows if r['split']==a.split and r['annotation_status']=='teacher_reviewed_episode'),
                key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:a.limit]
    if not rows:raise ValueError('No eligible held-out episodes')
    if any(r['image_sha256'] in train_images for r in rows):raise ValueError('Train image leakage')
    if not a.engineering_only and any(r.get('benchmark_dedup_status')!='cleared' for r in rows):
        raise ValueError('Benchmark clearance pending')
    backend=LatentBackend(a.model,root/rows[0]['image'],True,max_pixels=50176,
        adapter_path=a.adapter,resampler_path=a.resampler,quantize_vision=a.quantize_vision)
    policies={k:Policy(**v) for k,v in json.loads(Path(a.policy).read_text())['strata'].items()}
    out.mkdir(parents=True);records=[]
    with (out/'results.jsonl').open('w') as stream:
        for row in rows:
            result=dict(id=row['id'],split=row['split'],image_sha256=row['image_sha256'],question=row['question'],
                        reference=row['answer'],reference_steps=len(row['steps']),reads=[])
            for step in row['steps']:
                for index,atom in enumerate(step['atoms']):
                    if atom['kind']!='visual':continue
                    backend.set_image(root/row['image'])
                    if backend.image_digest!=row['image_sha256']:raise ValueError('Image mismatch')
                    boxes=tuple(Box(*b) for b in atom['region']['boxes'])
                    query={k:atom[k] for k in ['subject','predicate']};prompt=backend.prompt(query,boxes)
                    # prompt() expects Box instances and renders original pixel coordinates.
                    native=QwenBackend.generate(backend,prompt,boxes,max_tokens=32)[0]
                    compact=backend.propose(query,Sources(boxes,boxes,True),1)
                    compact=compact[0] if compact else ''
                    result['reads'].append(dict(step=step['id'],atom=index,reference=atom['value'],
                        native=native,compact=compact,native_correct=canonical(native)==canonical(atom['value']),
                        compact_correct=canonical(compact)==canonical(atom['value']),calls=list(backend.calls)))
            backend.set_image(root/row['image'])
            direct=backend.generate(row['question']+'\nReturn a short canonical answer.',
                [Box(0,0,backend.image.width,backend.image.height)],max_tokens=32)[0]
            result['direct']=dict(answer=direct,correct=canonical(direct)==canonical(row['answer']))
            backend.set_image(root/row['image'])
            audited=run(backend,row['question'],policies)
            (out/(row['id']+'-trace.json')).write_text(json.dumps(audited,indent=2))
            result['efa']=dict(final=audited['final'],budget=audited['budget'],
                correct=canonical(audited['final']['answer'])==canonical(row['answer']))
            records.append(result);stream.write(json.dumps(result)+'\n');stream.flush()
            print(json.dumps(dict(id=row['id'],direct=result['direct'],efa=result['efa'])),flush=True)
    reads=[r for row in records for r in row['reads']]
    summary=dict(kind='heldout_reviewed_episode_engineering_evaluation',count=len(records),split=a.split,
        visual_atoms=len(reads),native_correct=sum(r['native_correct'] for r in reads),
        compact_correct=sum(r['compact_correct'] for r in reads),direct_correct=sum(r['direct']['correct'] for r in records),
        efa_correct=sum(r['efa']['correct'] for r in records),verified=sum(r['efa']['final']['verified'] for r in records),
        generated_steps=[r['efa']['budget']['steps'] for r in records],checkpoint=backend.encoder_config,
        calibrated=False,benchmark_dedup_status='pending',reference_basis='native_annotations_and_teacher_checks',
        supplied_boxes_only_for_component_reads=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
