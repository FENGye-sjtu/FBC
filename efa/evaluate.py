"""Held-out component comparison and bounded EFA runs; no calibrated gate claims."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from efa.core import Policy
from efa.infer import run
from efa.latent_backend import LatentBackend
from efa.native import property_atom
from efa.qwen import QwenBackend,Sources
from efa.regions import Box


def canonical(value):
    return ' '.join(str(value).strip().casefold().rstrip('.').split())


def main():
    p=argparse.ArgumentParser()
    for arg in ['model','adapter','resampler','root','output','policy']:p.add_argument('--'+arg,required=True)
    p.add_argument('--split',choices=['dev','calibration'],default='dev')
    p.add_argument('--limit',type=int,default=16);p.add_argument('--episodes',type=int,default=2)
    p.add_argument('--engineering-only',action='store_true')
    p.add_argument('--quantize-vision',action='store_true',help='Historical NF4-vision checkpoint replay')
    a=p.parse_args();root=Path(a.root);out=Path(a.output)
    if out.exists():raise ValueError('Use a fresh evaluation directory')
    raw=[json.loads(s) for s in (root/'raw_tasks.jsonl').read_text().splitlines()]
    train_images={r['image_sha256'] for r in raw if r['split']=='train'}
    rows=[r for r in raw if r['split']==a.split and r.get('bbox') and property_atom(r['question'],r['answer'])]
    rows=sorted(rows,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:a.limit]
    if not rows or a.limit<1 or not 0<=a.episodes<=len(rows):raise ValueError('Invalid evaluation selection')
    if any(r['image_sha256'] in train_images for r in rows):raise ValueError('Exact image leakage')
    if not a.engineering_only and any(r.get('benchmark_dedup_status')!='cleared' for r in rows):
        raise ValueError('Benchmark clearance is pending')
    backend=LatentBackend(a.model,root/rows[0]['image'],True,max_pixels=50176,
                          adapter_path=a.adapter,resampler_path=a.resampler,quantize_vision=a.quantize_vision)
    policy=json.loads(Path(a.policy).read_text())
    policies={k:Policy(**v) for k,v in policy['strata'].items()}
    out.mkdir(parents=True)
    records=[];started=time.monotonic()
    with (out/'results.jsonl').open('w') as stream:
        for index,row in enumerate(rows):
            backend.set_image(root/row['image'])
            if backend.image_digest!=row['image_sha256']:raise ValueError('Image provenance mismatch')
            atom=property_atom(row['question'],row['answer'])
            query={k:atom[k] for k in ('subject','predicate')};box=Box(*row['bbox'])
            prompt=backend.prompt(query,[box]);source=Sources((box,),(box,),True)
            result=dict(id=row['id'],question=row['question'],reference=row['answer'],image_sha256=row['image_sha256'],
                        split=row['split'],source_box=row['bbox'],readers={})
            for mode in ['native_crop','compact_crop']:
                begin=time.monotonic();before=len(backend.calls)
                if mode=='native_crop':
                    predicted=QwenBackend.generate(backend,prompt,[box],max_tokens=32)[0]
                    logp,n=QwenBackend.logp(backend,prompt,[box],row['answer'])
                else:
                    candidates=backend.propose(query,source,1)
                    predicted=candidates[0] if candidates else ''
                    logp,n=backend.logp(prompt,[box],row['answer'])
                result['readers'][mode]=dict(answer=predicted,correct=canonical(predicted)==canonical(row['answer']),
                    reference_nll_per_token=-logp/n,seconds=time.monotonic()-begin,calls=backend.calls[before:])
            if index<a.episodes:
                backend.set_image(root/row['image'])
                direct=backend.generate(row['question']+'\nReturn a short canonical answer.',
                    [Box(0,0,backend.image.width,backend.image.height)],max_tokens=32)[0]
                result['direct']=dict(answer=direct,correct=canonical(direct)==canonical(row['answer']),calls=list(backend.calls))
                backend.set_image(root/row['image'])
                audited=run(backend,row['question'],policies)
                (out/(row['id']+'-episode.json')).write_text(json.dumps(audited,indent=2))
                result['efa']=dict(final=audited['final'],budget=audited['budget'],
                    correct=canonical(audited['final']['answer'])==canonical(row['answer']),
                    call_counts=dict(Counter(c['operation'] for c in audited['cost'])))
            stream.write(json.dumps(result)+'\n');stream.flush();records.append(result)
            print(json.dumps(dict(id=row['id'],readers={k:v['correct'] for k,v in result['readers'].items()},efa=result.get('efa'))),flush=True)
    summary=dict(scope='heldout_engineering_pilot_not_benchmark',split=a.split,count=len(records),
        checkpoint=backend.encoder_config,reader_decoding='greedy_max_new_tokens_32',native_boxes_supplied_to_reader_comparison=True,
        readers={mode:dict(correct=sum(r['readers'][mode]['correct'] for r in records),
            mean_reference_nll=sum(r['readers'][mode]['reference_nll_per_token'] for r in records)/len(records))
            for mode in ['native_crop','compact_crop']},
        full_image_cases=a.episodes,direct_correct=sum(r.get('direct',{}).get('correct',False) for r in records),
        efa_correct=sum(r.get('efa',{}).get('correct',False) for r in records),
        verified=sum(r.get('efa',{}).get('final',{}).get('verified',False) for r in records),
        seconds=time.monotonic()-started,calibrated=False,benchmark_dedup_status='pending')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))


if __name__=='__main__':main()
