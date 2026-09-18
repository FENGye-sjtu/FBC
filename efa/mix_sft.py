"""Combine eligible source exports while retaining per-source counts and splits."""
import argparse
import hashlib
import json
from pathlib import Path
from efa.export import export


def mix(sources,output,engineering_only=False):
    output=Path(output)
    if output.exists():raise ValueError('Use a fresh mixed export')
    train=set();heldout=set()
    for source in sources:
        for line in (Path(source)/'raw_tasks.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            (train if row['split']=='train' else heldout).add(row['image_sha256'])
    if train&heldout:raise ValueError('Train/held-out image collision across sources')
    output.mkdir(parents=True);combined={};info={};components=[]
    for index,source in enumerate(sources):
        destination=output/f'source_{index}'
        export(source,destination,engineering_only)
        descriptor=json.loads((destination/'dataset_info.json').read_text())
        for name,value in descriptor.items():
            combined.setdefault(name,[]).extend(json.loads((destination/value['file_name']).read_text(encoding='utf-8')))
            info[name]=value
        components.append(dict(source=str(Path(source).resolve()),
            manifest_sha256=hashlib.sha256((Path(source)/'manifest.json').read_bytes()).hexdigest(),
            counts=json.loads((destination/'export_manifest.json').read_text())['counts']))
    for name,rows in combined.items():
        (output/(name+'.json')).write_text(json.dumps(rows),encoding='utf-8')
    (output/'dataset_info.json').write_text(json.dumps(info,indent=2))
    manifest=dict(kind='mixed_program_sft_export',components=components,
        counts={k:len(v) for k,v in combined.items()},engineering_only=engineering_only,
        benchmark_dedup_override=engineering_only,source_images_cross_split_checked=True)
    (output/'export_manifest.json').write_text(json.dumps(manifest,indent=2));return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',action='append',required=True)
    p.add_argument('--output',required=True);p.add_argument('--engineering-only',action='store_true')
    a=p.parse_args();print(json.dumps(mix(a.source,a.output,a.engineering_only)))
