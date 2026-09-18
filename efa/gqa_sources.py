"""Add native GQA program-object region candidates without inventing step labels."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.request
import zipfile

URL='https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip'


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True)
    p.add_argument('--output',required=True);p.add_argument('--cache',required=True)
    a=p.parse_args();source=Path(a.source);out=Path(a.output);cache=Path(a.cache)
    if out.exists():raise ValueError('Choose a new data version')
    cache.mkdir(parents=True,exist_ok=True);archive=cache/'sceneGraphs.zip'
    if not archive.exists():
        part=cache/'sceneGraphs.zip.part'
        if not part.exists() or not zipfile.is_zipfile(part):
            with urllib.request.urlopen(URL,timeout=90) as response,part.open('wb') as output:
                shutil.copyfileobj(response,output)
        with zipfile.ZipFile(part) as z:
            if z.testzip() is not None:raise ValueError('Corrupt scene graph archive')
        part.replace(archive)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    rows=[json.loads(s) for s in (source/'raw_tasks.jsonl').read_text(encoding='utf-8').splitlines()]
    needed={Path(r['source_image_url']).stem for r in rows}
    graphs={}
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not name.endswith('.json'):continue
            with z.open(name) as stream:values=json.load(stream)
            graphs.update({key:values[key] for key in needed if key in values})
            del values
    out.mkdir(parents=True);(out/'images').mkdir()
    enriched=0
    for row in rows:
        graph=graphs.get(Path(row['source_image_url']).stem)
        ids=[m.group(1) for op in row['source_program'] if (m:=re.search(r'\((\d+)\)',op['argument']))]
        objects={}
        if graph:
            for oid in dict.fromkeys(ids):
                obj=graph['objects'].get(oid)
                if not obj:continue
                sx=row['width']/graph['width'];sy=row['height']/graph['height']
                box=[round(obj['x']*sx),round(obj['y']*sy),round((obj['x']+obj['w'])*sx),round((obj['y']+obj['h'])*sy)]
                box=[max(0,min(v,row['width'] if i%2==0 else row['height'])) for i,v in enumerate(box)]
                if box[0]>=box[2] or box[1]>=box[3]:continue
                objects[oid]=dict(bbox=box,name=obj['name'],attributes=obj['attributes'],relations=obj['relations'])
            if objects:
                row['native_candidate_boxes']=list({tuple(b):b for b in row['native_candidate_boxes']+[v['bbox'] for v in objects.values()]}.values())
                enriched+=1
        row['native_program_objects']=objects
        row['native_scene_graph_source']=dict(url=URL,sha256=digest)
        shutil.copy2(source/row['image'],out/row['image'])
    (out/'raw_tasks.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    manifest=json.loads((source/'manifest.json').read_text())
    manifest.update(native_program_region_tasks=enriched,native_scene_graph_sha256=digest,source_version=str(source))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest))


if __name__=='__main__':main()
