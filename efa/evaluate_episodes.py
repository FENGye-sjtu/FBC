"""Evaluate explicitly selected held-out annotation tiers and generated trajectories."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from efa.core import Policy
from efa.evaluate import canonical
from efa.infer import run
from efa.latent_backend import LatentBackend
from efa.qwen import QwenBackend,Sources
from efa.regions import Box
from efa.step_baseline import run_step_only


EPISODE_TIERS={
    'teacher_reviewed_episodes.jsonl':'teacher_reviewed_episode',
    'native_episodes.jsonl':'native_program_episode',
    'reviewed_episodes.jsonl':'reviewed_episode',
}


def select_episodes(root,episode_file,split,limit,engineering_only=False):
    """Keep the existing hash selection while enforcing each file's label tier."""
    if episode_file not in EPISODE_TIERS:raise ValueError('Unknown episode annotation file')
    if split not in ('dev','calibration') or limit<1:raise ValueError('Held-out split and positive limit required')
    status=EPISODE_TIERS[episode_file]
    if status!='reviewed_episode' and not engineering_only:
        raise ValueError('Native and teacher-reviewed labels require --engineering-only')
    root=Path(root)
    all_rows=[json.loads(s) for s in (root/episode_file).read_text(encoding='utf-8').splitlines() if s.strip()]
    # A train image remains a train image even if its row failed source review
    # or lives in another annotation tier of the same source corpus.
    train_images={r['image_sha256'] for r in all_rows if r['split']=='train'}
    for name in ('raw_tasks.jsonl',*EPISODE_TIERS):
        path=root/name
        if name==episode_file or not path.exists():continue
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            row=json.loads(line)
            if row['split']=='train':train_images.add(row['image_sha256'])
    rows=sorted((r for r in all_rows if r['split']==split and r.get('annotation_status')==status),
                key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:limit]
    if not rows:raise ValueError('No eligible held-out episodes')
    if any(r['image_sha256'] in train_images for r in rows):raise ValueError('Train image leakage')
    if not engineering_only and any(r.get('benchmark_dedup_status')!='cleared' for r in rows):
        raise ValueError('Benchmark clearance pending')
    return rows


def annotation_provenance(row,episode_file):
    return dict(episode_file=episode_file,annotation_status=row['annotation_status'],
        source_repo=row.get('source_repo'),source_revision=row.get('source_revision'),
        label_basis=row.get('label_basis'),review_label_basis=row.get('review',{}).get('label_basis'),
        benchmark_dedup_status=row.get('benchmark_dedup_status','pending'))


def reference_components(backend,root,row):
    """Read supplied visual sources and reference-conditioned text deductions.

    All prior reference claims are exposed, matching SFT and runtime ledger
    serialization. Unknown audit status stays None, as in the SFT exporter.
    These reads never feed the subsequent fully generated trajectories.
    """
    reads=[];premise_reads=[];context=[];provenance=[]
    for number,step in enumerate(row['steps'],1):
        for index,atom in enumerate(step['atoms']):
            backend.set_image(Path(root)/row['image'])
            if backend.image_digest!=row['image_sha256']:raise ValueError('Image mismatch')
            query={k:atom[k] for k in ('subject','predicate')}
            if atom['kind']=='visual':
                boxes=tuple(Box(*b) for b in atom['region']['boxes'])
                prompt=backend.prompt(query,boxes)
                native=QwenBackend.generate(backend,prompt,boxes,max_tokens=32)[0]
                compact=backend.propose(query,Sources(boxes,boxes,True),1)
                compact=compact[0] if compact else ''
                reads.append(dict(step=step['id'],atom=index,reference=atom['value'],
                    native=native,compact=compact,native_correct=canonical(native)==canonical(atom['value']),
                    compact_correct=canonical(compact)==canonical(atom['value']),calls=list(backend.calls)))
            elif atom['kind']=='premise':
                if not context:raise ValueError('Premise read requires earlier exposed claims')
                backend.audit_kind='premise';backend.premises=deepcopy(context)
                source=Sources((),(),True)
                candidates=backend.propose(query,source,1)
                answer=candidates[0] if candidates else ''
                premise_reads.append(dict(step=step['id'],atom=index,reference=atom['value'],answer=answer,
                    correct=canonical(answer)==canonical(atom['value']),calls=list(backend.calls),
                    conditioning='reference_prior_claims_not_generated_student_claims',
                    exposed_premises=deepcopy(context),inherited_reference_sources=deepcopy(provenance),
                    supplied_images=0,measured_audit=False))
            else:raise ValueError('Unsupported reference atom kind')
        # As in runtime, sibling atoms are not exposed before their boundary.
        # Annotated program dependencies are retained separately; actual prompt
        # exposure includes all earlier steps, not just those program parents.
        parents=[p['key'] for p in context]
        for index,atom in enumerate(step['atoms']):
            key=f'step{number}.atom{index}@1'
            claim={k:atom[k] for k in ('subject','predicate','value','family','kind')}
            context.append(dict(key=key,claim=dict(claim,parents=list(parents)),verified=None))
            provenance.append(dict(key=key,image_sha256=row['image_sha256'],kind=atom['kind'],
                region=deepcopy(atom.get('region')),native_evidence=deepcopy(atom.get('native_evidence')),
                annotated_parent_steps=list(step.get('parents',[])),exposed_parents=list(parents)))
    return reads,premise_reads


def verification_counts(records):
    verified=[r for r in records if r['efa']['final']['verified']]
    return dict(verified=len(verified),verified_correct=sum(r['efa']['correct'] for r in verified),
                verified_wrong=sum(not r['efa']['correct'] for r in verified))


def main():
    p=argparse.ArgumentParser()
    for key in ['model','adapter','resampler','root','output','policy']:p.add_argument('--'+key,required=True)
    p.add_argument('--split',choices=['dev','calibration'],default='dev')
    p.add_argument('--episode-file',choices=tuple(EPISODE_TIERS),default='teacher_reviewed_episodes.jsonl',
                   help='Explicit annotation tier; native and teacher labels require --engineering-only')
    p.add_argument('--limit',type=int,default=2);p.add_argument('--engineering-only',action='store_true')
    p.add_argument('--quantize-vision',action='store_true',help='Historical NF4-vision checkpoint replay')
    p.add_argument('--reacquire-on-binding-failure',action='store_true',
                   help='Experimental neutral source relocation after binding/invalid failure; default expands context')
    a=p.parse_args();root=Path(a.root);out=Path(a.output)
    if out.exists() or a.limit<1:raise ValueError('Fresh output and positive case limit required')
    rows=select_episodes(root,a.episode_file,a.split,a.limit,a.engineering_only)
    backend=LatentBackend(a.model,root/rows[0]['image'],True,max_pixels=50176,
        adapter_path=a.adapter,resampler_path=a.resampler,quantize_vision=a.quantize_vision,
        reacquire_on_binding_failure=a.reacquire_on_binding_failure)
    evaluation_settings=dict(reacquire_on_binding_failure=a.reacquire_on_binding_failure)
    policies={k:Policy(**v) for k,v in json.loads(Path(a.policy).read_text())['strata'].items()}
    out.mkdir(parents=True);records=[]
    with (out/'results.jsonl').open('w') as stream:
        for row in rows:
            result=dict(id=row['id'],split=row['split'],image_sha256=row['image_sha256'],question=row['question'],
                        reference=row['answer'],reference_steps=len(row['steps']),
                        evaluation_settings=evaluation_settings,
                        annotation_provenance=annotation_provenance(row,a.episode_file))
            result['reads'],result['reference_premise_reads']=reference_components(backend,root,row)
            backend.set_image(root/row['image'])
            direct=backend.generate(row['question']+'\nReturn a short canonical answer.',
                [Box(0,0,backend.image.width,backend.image.height)],max_tokens=32)[0]
            result['direct']=dict(answer=direct,correct=canonical(direct)==canonical(row['answer']),calls=list(backend.calls))
            backend.set_image(root/row['image'])
            step_only=run_step_only(backend,row['question'])
            (out/(row['id']+'-step-only.json')).write_text(json.dumps(step_only,indent=2))
            result['step_only']=dict(final=step_only['final'],budget=step_only['budget'],
                correct=canonical(step_only['final']['answer'])==canonical(row['answer']))
            backend.set_image(root/row['image'])
            audited=run(backend,row['question'],policies)
            audited['evaluation_settings']=evaluation_settings
            (out/(row['id']+'-trace.json')).write_text(json.dumps(audited,indent=2))
            result['efa']=dict(final=audited['final'],budget=audited['budget'],
                correct=canonical(audited['final']['answer'])==canonical(row['answer']))
            records.append(result);stream.write(json.dumps(result)+'\n');stream.flush()
            print(json.dumps(dict(id=row['id'],direct=result['direct'],efa=result['efa'])),flush=True)
    reads=[r for row in records for r in row['reads']]
    premises=[r for row in records for r in row['reference_premise_reads']]
    summary=dict(kind='heldout_reviewed_episode_engineering_evaluation' if a.episode_file!='native_episodes.jsonl'
                 else 'heldout_native_episode_engineering_evaluation',count=len(records),split=a.split,
        visual_atoms=len(reads),native_correct=sum(r['native_correct'] for r in reads),
        compact_correct=sum(r['compact_correct'] for r in reads),direct_correct=sum(r['direct']['correct'] for r in records),
        step_only_correct=sum(r['step_only']['correct'] for r in records),
        step_only_generated_steps=[r['step_only']['budget']['steps'] for r in records],
        efa_correct=sum(r['efa']['correct'] for r in records),**verification_counts(records),
        generated_steps=[r['efa']['budget']['steps'] for r in records],checkpoint=backend.encoder_config,
        evaluation_settings=evaluation_settings,
        calibrated=False,benchmark_dedup_status='cleared' if all(r.get('benchmark_dedup_status')=='cleared' for r in rows) else 'pending',
        reference_basis={'teacher_reviewed_episodes.jsonl':'native_annotations_and_teacher_checks',
                         'native_episodes.jsonl':'native_program_annotations',
                         'reviewed_episodes.jsonl':'reviewed_episode_annotations'}[a.episode_file],
        supplied_boxes_only_for_component_reads=True,engineering_only=a.engineering_only,
        episode_file=a.episode_file,episode_file_sha256=hashlib.sha256((root/a.episode_file).read_bytes()).hexdigest(),
        annotation_status_counts=dict(Counter(r['annotation_status'] for r in rows)),
        reference_premise_atoms=len(premises),reference_premise_correct=sum(r['correct'] for r in premises),
        premise_component_conditioning='reference_prior_claims_not_generated_student_claims',
        reference_claims_supplied_to_generated_trajectories=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
