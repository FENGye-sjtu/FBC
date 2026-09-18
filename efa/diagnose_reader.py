"""Attribute failures to the base model, SFT stage or source prompt on fixed cases."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from efa.evaluate import canonical
from efa.native import property_atom
from efa.qwen import QwenBackend
from efa.regions import Box


def main():
    p=argparse.ArgumentParser()
    for key in ['model','adapter','region-adapter','root','evaluation','output']:
        p.add_argument('--'+key,required=True)
    a=p.parse_args();root=Path(a.root);out=Path(a.output)
    if out.exists():raise ValueError('Choose a fresh diagnostic directory')
    selected=[json.loads(s)['id'] for s in Path(a.evaluation).read_text().splitlines()]
    source={r['id']:r for r in map(json.loads,(root/'raw_tasks.jsonl').read_text().splitlines())}
    rows=[source[k] for k in selected]
    if any(r['split']=='train' for r in rows):raise ValueError('Expected held-out cases')
    backend=QwenBackend(a.model,root/rows[0]['image'],True,max_pixels=50176,adapter_path=a.adapter)
    backend.model.load_adapter(a.region_adapter,adapter_name='region')
    backend.model.eval()
    out.mkdir(parents=True)
    results=[]
    with (out/'results.jsonl').open('w') as stream:
        for row in rows:
            atom=property_atom(row['question'],row['answer']);box=Box(*row['bbox'])
            result=dict(id=row['id'],reference=row['answer'],split=row['split'],conditions={})
            for stage in ['base','region','episode']:
                backend.model.set_adapter('region' if stage=='region' else 'default')
                with backend.model.disable_adapter() if stage=='base' else nullcontext():
                    for interface in ['neutral','question']:
                        backend.set_image(root/row['image'])
                        prompt=backend.prompt({k:atom[k] for k in ['subject','predicate']},[box]) if interface=='neutral' else row['question']+'\nReturn a short canonical answer.'
                        answer=backend.generate(prompt,[box],max_tokens=32)[0]
                        result['conditions'][stage+'_'+interface]=dict(answer=answer,correct=canonical(answer)==canonical(row['answer']))
            stream.write(json.dumps(result)+'\n');stream.flush();results.append(result)
            print(json.dumps(result),flush=True)
    summary=dict(kind='fixed_case_stage_diagnostic',count=len(rows),calibrated=False,
        benchmark_dedup_status='pending',max_pixels=50176,
        correct={k:sum(r['conditions'][k]['correct'] for r in results) for k in results[0]['conditions']},
        checkpoints={name:hashlib.sha256((Path(path)/'adapter_model.safetensors').read_bytes()).hexdigest()
                     for name,path in [('region',a.region_adapter),('episode',a.adapter)]})
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))


if __name__=='__main__':main()
