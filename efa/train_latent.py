"""Engineering alignment of the compact reader; no fabricated negative labels."""
import argparse
import json
from pathlib import Path
import random
import time
import torch
from PIL import Image
from efa.latent_backend import LatentBackend
from efa.regions import Box


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True);p.add_argument('--root',required=True)
    p.add_argument('--adapter');p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=40);p.add_argument('--engineering-only',action='store_true')
    a=p.parse_args();root=Path(a.root);out=Path(a.output)
    if out.exists():
        raise ValueError('Choose a new run directory')
    rows=[json.loads(s) for s in (root/'raw_tasks.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['split']=='train' and (a.engineering_only or r.get('benchmark_dedup_status')=='cleared')]
    if not rows or a.steps<1:
        raise ValueError('No eligible examples or invalid training budget')
    random.seed(2027);torch.manual_seed(2027)
    backend=LatentBackend(a.model,root/rows[0]['image'],True,max_pixels=50176,
                          adapter_path=a.adapter,allow_untrained=True)
    backend.model.requires_grad_(False)
    backend.resampler.train()
    optimizer=torch.optim.AdamW(backend.resampler.parameters(),lr=1e-4)
    out.mkdir(parents=True)
    losses=[];start=time.monotonic()
    for step in range(a.steps):
        row=rows[step%len(rows)]
        backend.image=Image.open(root/row['image']).convert('RGB')
        backend.features.clear()
        box=Box(*row['bbox'])
        prompt=row['question']+'\nReturn a short canonical answer.'
        optimizer.zero_grad(set_to_none=True)
        logp,n=backend.sequence_logp(prompt,[box],row['answer'])
        loss=-logp/n
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite reader loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(backend.resampler.parameters(),1.0)
        optimizer.step()
        losses.append(loss.item())
        print(json.dumps(dict(step=step+1,loss=losses[-1],task=row['id'])),flush=True)
    torch.save(dict(resampler=backend.resampler.state_dict(),steps=a.steps),out/'resampler.pt')
    metrics=dict(steps=a.steps,examples=len(rows),losses=losses,seconds=time.monotonic()-start,
                 engineering_only=a.engineering_only,scope='positive_region_alignment_smoke',
                 trained_objectives=['region_answer_ce'],calibrated=False,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics))


if __name__=='__main__':
    main()
