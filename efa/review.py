"""Fresh-context teacher checks; these remain pseudo-labels, not runtime passes."""
import argparse
import base64
import io
import json
from pathlib import Path
from PIL import Image
from efa.annotate import teacher_json

REVIEW_PROMPT='''Check only the supplied source crops or explicit premises.
Return JSON {"supported":true,"insufficient":false,"canonical_read":"short value","reason":"brief evidence"}.
Judge the supplied claim, not what an imagined full image might show. A partial,
ambiguous or wrongly bound source cannot support the claim. Do not assume the
claim value is correct. This is offline annotation review, not a measured VLM audit.'''


def canonical(value):
    return str(value).strip().casefold().rstrip('.')


def usable_sources(row):
    return all(a.get('predicate','').strip().lower()!='relation'
               for s in row['steps'] for a in s['atoms']) and all(a.get('region',{}).get('insufficient') is False
               for s in row['steps'] for a in s['atoms'] if a['kind']=='visual')


def review(row,root,cache):
    checks=[];steps={s['id']:s for s in row['steps']}
    original=Image.open(root/row['image']).convert('RGB')
    for step in row['steps']:
        for index,atom in enumerate(step['atoms']):
            claim={k:atom[k] for k in ['subject','predicate','value','kind']}
            if atom['kind']=='visual':
                region=atom['region']
                views={'target':list(dict.fromkeys(tuple(b) for b in region['boxes'])),
                       'expanded':list(dict.fromkeys(tuple(b) for b in region['boxes']+region['context_boxes']))}
                if views['target']==views['expanded']:views.pop('expanded')
            else:
                claim['premises']=[steps[p]['atoms'] for p in step['parents']]
                views={'premises':[]}
            for view,boxes in views.items():
                content=[]
                for box in boxes:
                    crop=original.crop(box)
                    if min(crop.size)<32:
                        crop=crop.resize((max(32,crop.width),max(32,crop.height)))
                    stream=io.BytesIO();crop.save(stream,format='PNG')
                    content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()}})
                payload=dict(claim)
                if boxes:payload['original_coordinates_in_image_order']=boxes
                content.append({'type':'text','text':json.dumps(payload)})
                result=teacher_json([{'role':'system','content':REVIEW_PROMPT},{'role':'user','content':content}],cache,'source_review_'+view)
                checks.append(dict(step=step['id'],atom=index,view=view,**result))
    reference_match=canonical(row['steps'][-1]['atoms'][-1]['value'])==canonical(row['answer'])
    passed=reference_match and usable_sources(row) and all(c['output'].get('supported') is True and c['output'].get('insufficient') is False for c in checks)
    return dict(row,annotation_status='teacher_reviewed_episode' if passed else 'teacher_review_rejected',
                review=dict(reference_match=reference_match,checks=checks,
                            label_basis='separate_target_and_expanded_teacher_checks_not_ground_truth'),runtime_audit=None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--limit',type=int,default=8)
    a=p.parse_args();root=Path(a.root);cache=root/'teacher_cache'
    rows=[json.loads(s) for s in (root/'pseudo_episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    destination=root/'teacher_reviewed_episodes.jsonl'
    existing={json.loads(s)['id'] for s in destination.read_text(encoding='utf-8').splitlines()} if destination.exists() else set()
    with destination.open('a',encoding='utf-8') as out:
        for row in rows[:a.limit]:
            if row['id'] in existing:continue
            result=review(row,root,cache)
            out.write(json.dumps(result,ensure_ascii=False)+'\n');out.flush()
            print(row['id'],result['annotation_status'],flush=True)


if __name__=='__main__':
    main()
