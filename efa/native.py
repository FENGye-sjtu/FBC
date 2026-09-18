"""Conservative conversion of explicit property questions into neutral atoms."""
import re
import json
from pathlib import Path


def property_atom(question,value):
    patterns=[r'^What is the (?:primary )?(color|shape|material) of (.+)\?$',
              r'^What (color|shape|material) is (.+)\?$']
    match=next((m for pattern in patterns if (m:=re.match(pattern,question))),None)
    if not match:
        return None
    predicate,subject=match.groups()
    # Do not turn a disputed color into a router identity.
    if predicate=='color' and re.search(r'\b(red|blue|green|yellow|pink|purple|black|white|brown|orange|beige)\b',subject,re.I):
        return None
    return dict(subject=subject,predicate=predicate,value=value,family='attribute',kind='visual')


def training_reads(root,engineering_only=False,episode_file=None):
    """Load only training atoms from the requested annotation tier."""
    root=Path(root);reads=[]
    if episode_file not in (None,'native_episodes.jsonl','teacher_reviewed_episodes.jsonl','reviewed_episodes.jsonl'):
        raise ValueError('Unknown episode annotation tier')
    rows=map(json.loads,(root/(episode_file or 'raw_tasks.jsonl')).read_text(encoding='utf-8').splitlines())
    for row in rows:
        if row['split']!='train' or (not engineering_only and row.get('benchmark_dedup_status')!='cleared'):
            continue
        if episode_file:
            expected={'native_episodes.jsonl':'native_program_episode',
                      'teacher_reviewed_episodes.jsonl':'teacher_reviewed_episode',
                      'reviewed_episodes.jsonl':'reviewed_episode'}[episode_file]
            if row.get('annotation_status')!=expected or (expected!='reviewed_episode' and not engineering_only):continue
            for step in row['steps']:
                for index,atom in enumerate(step['atoms']):
                    if atom['kind']!='visual' or atom['region'].get('insufficient') is not False:continue
                    reads.append(dict(id=f"{row['id']}-s{step['id']}-a{index}",episode_id=row['id'],image=row['image'],
                        query={k:atom[k] for k in ['subject','predicate']},answer=atom['value'],boxes=atom['region']['boxes']))
        elif row.get('bbox') and (atom:=property_atom(row['question'],row['answer'])):
            reads.append(dict(id=row['id'],episode_id=row['id'],image=row['image'],
                query={k:atom[k] for k in ['subject','predicate']},answer=row['answer'],boxes=[row['bbox']]))
    return reads
