"""Compile native GQA chains into source-attributed steps, without runtime PASS labels."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import urllib.error
import urllib.request
import zipfile
from PIL import Image
from efa.data import partition
from efa.regions import Box, complement
from efa.viscot import REVISION


def normalized(value):
    return ' '.join(value.casefold().strip().split())


def compile_episode(row):
    objects=row['native_program_objects'];steps=[];states={};referents={};anchors={}
    def obj(oid):
        item=objects[oid]
        if not Box(*item['bbox']).valid(row['width'],row['height']):
            raise ValueError('invalid_native_box')
        return item
    for index,operation in enumerate(row['source_program']):
        kind=operation['operation'];arg=operation['argument'];deps=operation['dependencies']
        if (index==0 and (kind!='select' or deps)) or (index>0 and deps!=[index-1]):
            raise ValueError('unsupported_non_linear_program')
        previous=states[deps[0]] if deps else None
        prior_ref=referents[deps[0]] if deps else None
        prior_anchors=anchors[deps[0]] if deps else []
        match=re.search(r'\((\d+)\)$',arg)
        oid=match.group(1) if match else previous
        if oid not in objects:raise ValueError('missing_program_object')
        item=obj(oid);boxes=[item['bbox']];family='identity'
        provenance=dict(operation_index=index,operation=operation,object_ids=[oid])
        if kind=='select':
            if match is None:raise ValueError('missing_selection_target')
            current_ref=arg[:match.start()].strip()
            subject=current_ref;predicate='is present in image';value='yes'
            provenance['field']='program_selection_and_native_object'
        elif kind=='relate':
            if match is None:raise ValueError('missing_relation_target')
            fields=arg[:match.start()].strip().split(',')
            if len(fields)!=3 or fields[2] not in ['s','o']:
                raise ValueError('unsupported_relation_direction')
            relation=fields[1]
            source,target=(oid,previous) if fields[2]=='s' else (previous,oid)
            if not any(r['object']==target and r['name']==relation for r in obj(source)['relations']):
                raise ValueError('missing_native_relation_edge')
            category=fields[0] if fields[0]!='_' else 'object'
            subject=category if fields[2]=='s' else prior_ref
            predicate='relationship to '+(prior_ref if fields[2]=='s' else category)
            value=relation;family='relation'
            current_ref=(f'{category} {relation} {prior_ref}' if fields[2]=='s'
                         else f'{category} that {prior_ref} is {relation}')
            boxes=[obj(source)['bbox'],obj(target)['bbox']]
            provenance.update(object_ids=[source,target],edge=dict(source=source,target=target,name=relation))
        elif kind.startswith('filter '):
            predicate=kind.removeprefix('filter ');value=arg;subject=prior_ref;family='attribute'
            if predicate in ['hposition','vposition']:
                b=item['bbox'];axis=0 if predicate=='hposition' else 1
                center=(b[axis]+b[axis+2])/2
                extent=row['width'] if axis==0 else row['height']
                expected=('left' if center<extent/2 else 'right') if axis==0 else ('top' if center<extent/2 else 'bottom')
                if arg!=expected:raise ValueError('position_disagrees_with_geometry')
                predicate='horizontal half of original image' if axis==0 else 'vertical half of original image'
                provenance['field']='native_box_center_relative_to_frame'
            else:
                if arg not in item['attributes']:raise ValueError('missing_native_filter_attribute')
                provenance['field']='attributes'
            current_ref=f'{prior_ref} with {predicate} {value}'
        elif kind=='query':
            if index!=len(row['source_program'])-1:raise ValueError('nonterminal_query')
            subject=prior_ref;predicate='category' if arg=='name' else arg;current_ref=prior_ref
            if arg=='name':value=item['name']
            elif arg in ['color','material','shape','size','texture'] and row['answer'] in item['attributes']:
                value=row['answer'];family='attribute'
            else:raise ValueError('unsupported_or_unannotated_query')
            provenance['field']='name' if arg=='name' else 'attributes'
        else:raise ValueError('unsupported_operation:'+kind)
        # A composite referent must bring its actual anchors to the target read,
        # not rely on a later whole-image context to hide missing source evidence.
        boxes=[list(b) for b in dict.fromkeys(tuple(b) for b in boxes+[obj(key)['bbox'] for key in prior_anchors])]
        atom=dict(subject=subject,predicate=predicate,value=value,family=family,kind='visual',
            region=dict(boxes=boxes,context_boxes=boxes+[[0,0,row['width'],row['height']]],
                        insufficient=False,coordinate_system='original_pixels',
                        label_basis='native_scene_graph_not_independent_visual_review'),
            complement_boxes=[list(vars(b).values()) for b in complement(row['width'],row['height'],[Box(*b) for b in boxes])],
            native_evidence=provenance)
        steps.append(dict(id=index+1,text=f'Step {index+1}: {subject} / {predicate}: {value}',
                          parents=[d+1 for d in deps],atoms=[atom]))
        states[index]=oid
        referents[index]=current_ref
        anchors[index]=list(dict.fromkeys([oid]+prior_anchors))
    if not 3<=len(steps)<=8 or row['source_program'][-1]['operation']!='query':
        raise ValueError('unsupported_episode_length_or_terminal')
    if normalized(steps[-1]['atoms'][-1]['value'])!=normalized(row['answer']):
        raise ValueError('final_answer_disagrees_with_native_annotation')
    return dict(row,steps=steps,annotation_status='native_program_episode',
        label_basis='GQA_native_program_and_scene_graph_consistency_not_runtime_verification',runtime_audit=None)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--metadata');p.add_argument('--graphs');p.add_argument('--source',help='Recompile existing original images and programs')
    p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=512)
    p.add_argument('--min-steps',type=int,choices=range(3,9),default=3)
    a=p.parse_args();out=Path(a.output)
    if out.exists() or a.limit<1:raise ValueError('Choose a new dataset directory and positive limit')
    if a.source:
        if a.min_steps!=3:p.error('Source recompilation preserves its existing selection; use metadata for a length-stratified sample')
        import shutil
        source=Path(a.source)
        records=[compile_episode(json.loads(s)) for s in (source/'raw_tasks.jsonl').read_text(encoding='utf-8').splitlines()]
        out.mkdir(parents=True);(out/'images').mkdir()
        for row in records:shutil.copyfile(source/row['image'],out/row['image'])
        (out/'native_episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        shutil.copyfile(source/'raw_tasks.jsonl',out/'raw_tasks.jsonl')
        manifest=json.loads((source/'manifest.json').read_text())
        manifest.update(parent_version=str(source),
            compiler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            target_source_policy='target_and_all_referent_anchors')
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
        return
    if not a.metadata or not a.graphs:p.error('Supply --source or both --metadata and --graphs')
    metadata=Path(a.metadata);graphs_path=Path(a.graphs)
    candidates=[]
    for index,line in enumerate(metadata.open(encoding='utf-8')):
        row=json.loads(line)
        if row['split']=='train' and a.min_steps<=len(row.get('reasoning',[]))<=8:
            candidates.append((index,row))
    random.Random(20260920).shuffle(candidates)
    graphs={}
    with zipfile.ZipFile(graphs_path) as z:
        for name in z.namelist():
            if name.endswith('train_sceneGraphs.json'):
                with z.open(name) as stream:graphs=json.load(stream)
    if not graphs:raise ValueError('Native train scene graphs unavailable')
    selected=[];rejected=Counter();seen=set()
    graph_sha=hashlib.sha256(graphs_path.read_bytes()).hexdigest()
    for index,raw in candidates:
        image_id=Path(raw['image']).stem
        if image_id in seen:continue
        graph=graphs.get(image_id)
        if graph is None:rejected['missing_train_graph']+=1;continue
        width,height=graph['width'],graph['height']
        ids={m.group(1) for op in raw['reasoning'] if (m:=re.search(r'\((\d+)\)$',op['argument']))}
        objects={}
        for oid in ids:
            if oid not in graph['objects']:continue
            item=graph['objects'][oid]
            box=[max(0,math.floor(item['x'])),max(0,math.floor(item['y'])),
                 min(width,math.ceil(item['x']+item['w'])),min(height,math.ceil(item['y']+item['h']))]
            objects[oid]=dict(bbox=box,name=item['name'],attributes=item['attributes'],relations=item['relations'])
        row=dict(id=f'gqa-native-{index}',source_repo='deepcs233/Visual-CoT',source_revision=REVISION,
            source_split='train',source_image_name=raw['image'],question=raw['question'],answer=raw['answer'],
            source_program=raw['reasoning'],native_program_objects=objects,width=width,height=height,
            native_scene_graph_source=dict(sha256=graph_sha,member='train_sceneGraphs.json'),
            benchmark_dedup_status='pending')
        try:episode=compile_episode(row)
        except ValueError as error:rejected[str(error)]+=1;continue
        selected.append(episode);seen.add(image_id)
        if len(selected)>=a.limit:break
    del graphs
    out.mkdir(parents=True);(out/'images').mkdir()
    def fetch(row):
        name=row['source_image_name']
        if Path(name).name!=name:raise ValueError('Invalid source image filename')
        for folder in ['VG_100K','VG_100K_2']:
            url=f'https://cs.stanford.edu/people/rak248/{folder}/{name}'
            try:
                with urllib.request.urlopen(url,timeout=60) as response:raw=response.read()
                break
            except urllib.error.HTTPError as error:
                if error.code!=404:raise
        else:raise ValueError('Original image unavailable')
        image=Image.open(io.BytesIO(raw)).convert('RGB')
        if image.size!=(row['width'],row['height']):raise ValueError('Native image frame mismatch')
        sha=hashlib.sha256(image.tobytes()+str(image.size).encode()).hexdigest()
        path='images/'+sha+'.jpg';(out/path).write_bytes(raw)
        return dict(row,image=path,image_sha256=sha,source_image_url=url,split=partition(sha))
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool,(out/'native_episodes.jsonl').open('w',encoding='utf-8') as stream:
        for row in pool.map(fetch,selected):
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();records.append(row)
            if len(records)%32==0:print('native_episodes',len(records),flush=True)
    (out/'raw_tasks.jsonl').write_text(''.join(json.dumps({k:v for k,v in r.items() if k!='steps'},ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    manifest=dict(kind='native_program_episodes',count=len(records),images=len({r['image_sha256'] for r in records}),
        splits=dict(Counter(r['split'] for r in records)),steps=sum(len(r['steps']) for r in records),
        rejected_programs=dict(rejected),source_revision=REVISION,metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
        scene_graph_sha256=graph_sha,compiler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        construction_complete=True,benchmark_dedup_status='pending',runtime_verified_episodes=0,
        min_program_steps=a.min_steps,step_lengths=dict(Counter(len(r['steps']) for r in records)),
        independent_visual_review=False,image_edits=False,sampling_seed=20260920,
        license_note='Visual-CoT card: YAML Apache-2.0; body CC-BY-NC-4.0; research use, original image rights retained.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
