"""Engineering alignment of the compact reader; no fabricated negative labels."""
import argparse
import json
from pathlib import Path
import random
import time
import torch
from efa.latent_backend import LatentBackend
from efa.regions import Box
from efa.reader_data import EPISODE_TIERS, load_training_sources, shuffled_reads, set_training_image
from efa.format import read_prompt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True);p.add_argument('--root',required=True)
    p.add_argument('--adapter');p.add_argument('--output',required=True)
    p.add_argument('--resampler',help='Optional existing compact reader to align with a new adapter')
    p.add_argument('--episode-file',choices=list(EPISODE_TIERS))
    p.add_argument('--episode-source',action='append',nargs=2,default=[],metavar=('ROOT','FILE'),
                   help='Additional explicit episode source; repeat to mix more datasets')
    p.add_argument('--seed',type=int,default=2027)
    p.add_argument('--quantize-vision',action='store_true',help='Historical all-module NF4 replay only')
    p.add_argument('--steps',type=int,default=40);p.add_argument('--engineering-only',action='store_true')
    a=p.parse_args();root=Path(a.root);out=Path(a.output)
    if out.exists():
        raise ValueError('Choose a new run directory')
    rows,data_manifest=load_training_sources(root,a.episode_file,a.episode_source,a.engineering_only)
    if not rows or a.steps<1:
        raise ValueError('No eligible examples or invalid training budget')
    random.seed(a.seed);torch.manual_seed(a.seed)
    backend=LatentBackend(a.model,rows[0]['image_path'],True,max_pixels=50176,
                          adapter_path=a.adapter,resampler_path=a.resampler,allow_untrained=True,quantize_vision=a.quantize_vision)
    backend.model.requires_grad_(False)
    backend.resampler.train()
    optimizer=torch.optim.AdamW(backend.resampler.parameters(),lr=1e-4)
    out.mkdir(parents=True)
    (out/'input_sources.json').write_text(json.dumps(data_manifest,indent=2),encoding='utf-8')
    losses=[];start=time.monotonic()
    for step,row in enumerate(shuffled_reads(rows,a.steps,a.seed)):
        set_training_image(backend,row)
        boxes=[Box(*b) for b in row['boxes']]
        prompt=read_prompt(row['query'],row['boxes'])
        optimizer.zero_grad(set_to_none=True)
        logp,n=backend.sequence_logp(prompt,boxes,row['answer'])
        loss=-logp/n
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite reader loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(backend.resampler.parameters(),1.0)
        optimizer.step()
        losses.append(loss.item())
        print(json.dumps(dict(step=step+1,loss=losses[-1],task=row['id'],source_index=row['source_index'])),flush=True)
    torch.save(dict(resampler=backend.resampler.state_dict(),steps=a.steps,encoder_config=backend.encoder_config),out/'resampler.pt')
    metrics=dict(steps=a.steps,examples=len(rows),losses=losses,seconds=time.monotonic()-start,
                 encoder_config=backend.encoder_config,
                 episode_file=a.episode_file,episodes=len({r['episode_id'] for r in rows}),
                 input_sources=data_manifest['sources'],seed=a.seed,sampling='shuffle_all_sources_each_epoch',
                 source_images_cross_split_checked=data_manifest['source_images_cross_split_checked'],
                 annotation_statuses_preserved=True,
                 adapter=a.adapter,initial_resampler=a.resampler,
                 engineering_only=a.engineering_only,scope='positive_region_alignment_smoke',
                 trained_objectives=['region_answer_ce'],calibrated=False,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics))


if __name__=='__main__':
    main()
