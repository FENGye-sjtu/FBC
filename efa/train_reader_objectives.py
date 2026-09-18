"""Native read/identity/unlikelihood training; optionally update shared LoRA."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from efa.latent_backend import LatentBackend
from efa.reader_objectives import load_objectives,backward_objectives


def main():
    p=argparse.ArgumentParser()
    for name in ['model','adapter','root','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--resampler');p.add_argument('--steps',type=int,default=768)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--adapter-learning-rate',type=float,default=1e-5)
    p.add_argument('--identity-weight',type=float,default=0.2)
    p.add_argument('--negative-weight',type=float,default=1.0)
    p.add_argument('--train-adapter',action='store_true')
    p.add_argument('--engineering-only',action='store_true')
    a=p.parse_args();root=Path(a.root);out=Path(a.output)
    if out.exists() or a.steps<1:raise ValueError('Fresh output and positive training budget required')
    if min(a.identity_weight,a.negative_weight)<0 or min(a.learning_rate,a.adapter_learning_rate)<=0:
        raise ValueError('Invalid loss weights or learning rates')
    rows=load_objectives(root,a.engineering_only)
    random.seed(2027);torch.manual_seed(2027)
    backend=LatentBackend(a.model,root/rows[0]['image'],True,max_pixels=50176,
        adapter_path=a.adapter,resampler_path=a.resampler,allow_untrained=True)
    backend.model.requires_grad_(False);backend.resampler.train()
    parameters=list(backend.resampler.parameters())
    groups=[dict(params=list(parameters),lr=a.learning_rate)]
    if a.train_adapter:
        lora=[]
        for name,param in backend.model.named_parameters():
            if '.lora_' in name:param.requires_grad_(True);lora.append(param)
        if not lora:raise ValueError('No shared LoRA parameters found')
        backend.model.train();backend.visual.eval()
        backend.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        groups.append(dict(params=lora,lr=a.adapter_learning_rate));parameters+=lora
    optimizer=torch.optim.AdamW(groups)
    out.mkdir(parents=True);start=time.monotonic();losses=[];order=list(range(len(rows)))
    with (out/'losses.jsonl').open('w') as stream:
        for step in range(a.steps):
            if step%len(rows)==0:random.shuffle(order)
            row=rows[order[step%len(rows)]];backend.set_image(root/row['image'])
            if backend.image_digest!=row['image_sha256']:raise ValueError('Image provenance mismatch')
            optimizer.zero_grad(set_to_none=True)
            metrics=backward_objectives(backend,row,a.identity_weight,a.negative_weight)
            torch.nn.utils.clip_grad_norm_(parameters,1.0);optimizer.step()
            record=dict(step=step+1,id=row['id'],**metrics)
            stream.write(json.dumps(record)+'\n');stream.flush();losses.append(metrics['total'])
            print(json.dumps(record),flush=True)
    if a.train_adapter:
        adapter=out/'adapter';backend.model.save_pretrained(adapter,safe_serialization=True)
        backend.encoder_config.update(adapter=str(adapter),
            adapter_sha256=hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest())
    torch.save(dict(resampler=backend.resampler.state_dict(),steps=a.steps,encoder_config=backend.encoder_config),out/'resampler.pt')
    metrics=dict(steps=a.steps,examples=len(rows),negative_atoms=sum(bool(r['negatives']) for r in rows),
        mean_loss=sum(losses)/len(losses),seconds=time.monotonic()-start,
        encoder_config=backend.encoder_config,config=vars(a),joint_adapter=a.train_adapter,
        objectives=['summed_sequence_read_ce']+(['identity_sequence_ce'] if a.identity_weight else [])+
            (['whole_sequence_unlikelihood'] if a.negative_weight else []),
        data_sha256=hashlib.sha256((root/'reader_objectives.jsonl').read_bytes()).hexdigest(),
        calibrated=False,scope='native_reader_objectives_engineering',
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics),flush=True)


if __name__=='__main__':main()
