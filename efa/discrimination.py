"""Native exclusive-class negatives, never inferred from missing annotations."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

# Deliberately exclude overlapping roles (man/driver), subclasses (car/taxi),
# materials, and relations that may coexist. This policy is an explicit textual
# incompatibility assumption on a native single-object category annotation.
EXCLUSIVE_GROUPS={
    'animal_species': ('cat','dog','cow','horse','zebra','elephant','giraffe'),
    'vehicle_kind': ('airplane','bicycle','bus','car','motorcycle','train'),
}
ALIASES={'airplanes':'airplane','cows':'cow'}
POLICY='native_exclusive_category_v1'


def canonical_class(value):
    value=' '.join(value.lower().strip().split())
    return ALIASES.get(value,value)


def category_negatives(episode,atom,limit=2):
    if atom['predicate']!='category' or atom['kind']!='visual':return []
    evidence=atom.get('native_evidence',{})
    if evidence.get('field')!='name' or len(evidence.get('object_ids',[]))!=1:return []
    oid=evidence['object_ids'][0]
    native=episode.get('native_program_objects',{}).get(oid)
    if not native or canonical_class(native['name'])!=canonical_class(atom['value']):return []
    positive=canonical_class(atom['value'])
    group=next((name for name,values in EXCLUSIVE_GROUPS.items() if positive in values),None)
    if group is None:return []
    candidates=[value for value in EXCLUSIVE_GROUPS[group] if value!=positive]
    candidates.sort(key=lambda value:hashlib.sha256((episode['id']+'|'+oid+'|'+value).encode()).hexdigest())
    return [dict(value=value,label_basis='native_single_object_category_and_explicit_exclusivity',
                 policy=POLICY,exclusive_group=group,positive_native_name=native['name'],object_id=oid)
            for value in candidates[:limit]]


def build(source,output):
    source=Path(source);output=Path(output)
    if output.exists():raise ValueError('Use a new dataset version')
    rows=[json.loads(s) for s in (source/'teacher_reviewed_episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    records=[]
    for episode in rows:
        # Held-out groups never contribute examples, negative vocabulary statistics
        # or repeated variants to this training-only artifact.
        if episode['split']!='train' or episode['annotation_status']!='teacher_reviewed_episode':continue
        for step in episode['steps']:
            for index,atom in enumerate(step['atoms']):
                if atom['kind']!='visual' or atom['region'].get('insufficient') is not False:continue
                boxes=atom['region']['boxes']
                context=[list(b) for b in dict.fromkeys(tuple(b) for b in boxes+atom['region']['context_boxes'])]
                records.append(dict(id=f"{episode['id']}-s{step['id']}-a{index}",episode_id=episode['id'],
                    image=episode['image'],image_sha256=episode['image_sha256'],split='train',
                    query={k:atom[k] for k in ['subject','predicate']},answer=atom['value'],
                    boxes=boxes,context_boxes=context,identity=atom['subject'],
                    negatives=category_negatives(episode,atom),native_evidence=atom['native_evidence'],
                    native_program_objects=episode['native_program_objects'],
                    positive_label_basis='native_program_with_teacher_source_checks',
                    benchmark_dedup_status=episode.get('benchmark_dedup_status','pending')))
    if not records or not any(r['negatives'] for r in records):raise ValueError('No eligible native negatives')
    output.mkdir(parents=True);(output/'images').mkdir()
    for image in sorted({r['image'] for r in records}):shutil.copyfile(source/image,output/image)
    (output/'reader_objectives.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    manifest=dict(kind='native_reader_objectives',count=len(records),images=len({r['image_sha256'] for r in records}),
        episodes=len({r['episode_id'] for r in records}),negative_atoms=sum(bool(r['negatives']) for r in records),
        negative_strings=sum(len(r['negatives']) for r in records),exclusive_groups=EXCLUSIVE_GROUPS,
        negative_distribution=dict(Counter(n['value'] for r in records for n in r['negatives'])),
        positive_distribution=dict(Counter(r['answer'] for r in records if r['negatives'])),
        policy=POLICY,source_sha256=hashlib.sha256((source/'teacher_reviewed_episodes.jsonl').read_bytes()).hexdigest(),
        construction_complete=True,engineering_only=True,benchmark_dedup_status='pending',
        teacher_labels_are_ground_truth=False,student_errors=False,image_edits=False,
        limitation='Explicit exclusive native categories only; no absent-label, relation, role or material negatives')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(build(a.source,a.output)))
