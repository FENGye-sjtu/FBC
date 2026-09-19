"""Select a balanced metadata-only batch for later individual visual review."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


SOURCES = {
    'zwz': Path('data/zwz_pilot_128_v1'),
    'deepeyes': Path('data/deepeyesv2_mixed_local_v1'),
    'gqa': Path('data/gqa_native_review_v2_complete'),
}
SPLIT_TARGETS = {
    'zwz': {'train': 12, 'dev': 2, 'calibration': 2},
    'deepeyes': {'train': 13, 'dev': 2, 'calibration': 1},
    'gqa': {'train': 11, 'dev': 2, 'calibration': 3},
}


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--exclude',action='append',default=[])
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError('Preserve the existing selection')
    excluded=set();registries=[]
    for root in map(Path,a.exclude):
        for name in ('raw_tasks.jsonl','reviewed_episodes.jsonl'):
            path=root/name
            if not path.is_file():continue
            payload=path.read_bytes();excluded.update(r['image_sha256'] for r in rows(path))
            registries.append(dict(path=str(path),sha256=hashlib.sha256(payload).hexdigest()))
    selected=[];seen=set()
    for family,root in SOURCES.items():
        raw=rows(root/'raw_tasks.jsonl');index={r['id']:r for r in raw}
        if family=='gqa':
            approved=[r for r in rows(root/'teacher_reviewed_episodes.jsonl') if r.get('annotation_status')=='teacher_reviewed_episode']
            allowed={r['id'] for r in approved}
            source_note='Previously teacher-reviewed native program; all visual labels still require fresh Codex review.'
        else:
            allowed=set(index)
            source_note='Source QA only; upstream answer and any prior trace are unverified hints.'
        available=[r for r in raw if r['id'] in allowed and r['image_sha256'] not in excluded|seen
                   and (root/r['image']).is_file()]
        for split,count in SPLIT_TARGETS[family].items():
            candidates=[r for r in available if r['split']==split]
            candidates.sort(key=lambda r:hashlib.sha256(f'efa-next48-v1:{family}:{r["id"]}'.encode()).hexdigest())
            if len(candidates)<count:raise ValueError(f'Not enough {family} {split} candidates')
            for row in candidates[:count]:
                seen.add(row['image_sha256'])
                selected.append(dict(id=row['id'],source_root=str(root),image=row['image'],
                    image_sha256=row['image_sha256'],split=row['split'],source_split=row.get('source_split'),
                    question=row['question'],upstream_answer_unverified=row['answer'],
                    candidate_type=family+'_fresh_visual_episode',selection_reason=source_note,
                    image_available_locally=True,pixel_hash_verified=True,visual_review_completed=False,
                    annotation_status='metadata_candidate_not_approved',benchmark_dedup_status='pending'))
    if len(selected)!=48 or len(seen)!=48:raise ValueError('Expected 48 distinct image groups')
    manifest=dict(kind='local_metadata_selection_not_approved',candidate_count=len(selected),
        source_counts=dict(Counter(r['source_root'] for r in selected)),
        split_counts=dict(Counter(r['split'] for r in selected)),excluded_source_files=registries,
        excluded_image_hash_count=len(excluded),exact_image_overlap_with_existing_batches=0,
        intra_selection_image_duplicates=0,source_split_policy='Preserve source split; group by decoded image before variants.',
        selection_scope='Deterministic balanced metadata selection; no visual approval.',
        teacher_policy='Qwen drafts only; Codex must inspect every original and final crop before acceptance.',
        benchmark_dedup_status='pending',candidates=selected)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='candidates'}))


if __name__=='__main__':main()
