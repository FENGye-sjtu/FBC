"""Read-only check of precision and image resolution on one identified source case."""
import argparse
import gc
import json
from pathlib import Path
import torch
from efa.qwen import QwenBackend
from efa.regions import Box


def main():
    p=argparse.ArgumentParser()
    for key in ['model','root','id','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--mixed-only',action='store_true')
    a=p.parse_args();root=Path(a.root);out=Path(a.output)
    if out.exists():raise ValueError('Choose a fresh probe directory')
    row=next(r for r in map(json.loads,(root/'raw_tasks.jsonl').read_text().splitlines()) if r['id']==a.id)
    out.mkdir(parents=True);records=[]
    with (out/'results.jsonl').open('w') as stream:
        for quantized,quantize_vision in ([(True,False)] if a.mixed_only else [(True,True),(True,False),(False,False)]):
            for pixels in [50176,200704]:
                torch.cuda.reset_peak_memory_stats()
                b=QwenBackend(a.model,root/row['image'],quantized,max_pixels=pixels,quantize_vision=quantize_vision)
                full=[Box(0,0,b.image.width,b.image.height)]
                answers={name:b.generate(prompt,full,max_tokens=96)[0] for name,prompt in
                    [('question',row['question']+'\nReturn a short canonical answer.'),
                     ('description','Describe the main vehicle in this image.') ]}
                record=dict(id=row['id'],reference=row['answer'],quantized=quantized,max_pixels=pixels,
                    quantize_vision=quantize_vision,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    answers=answers,vision_linear_types=sorted({type(module).__name__ for module in b.model.visual.modules() if 'Linear' in type(module).__name__}),
                    language_linear_types=sorted({type(module).__name__ for module in b.model.model.modules() if 'Linear' in type(module).__name__}),
                    pixel_digest=b.image_digest,calls=b.calls)
                records.append(record);stream.write(json.dumps(record)+'\n');stream.flush();print(json.dumps(record),flush=True)
                del b;gc.collect();torch.cuda.empty_cache()
    (out/'summary.json').write_text(json.dumps(dict(kind='precision_resolution_diagnostic',count=1,conditions=len(records),
        source_id=row['id'],reference=row['answer'],results=records),indent=2))


if __name__=='__main__':main()
