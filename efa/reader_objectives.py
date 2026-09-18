"""Paper sequence losses with explicit source-relative training provenance."""
import json
from pathlib import Path
from efa.discrimination import EXCLUSIVE_GROUPS,POLICY,canonical_class
from efa.format import read_prompt,binding_prompt
from efa.regions import Box


def load_objectives(root,engineering_only=False):
    root=Path(root);rows=[]
    for line in (root/'reader_objectives.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if row['split']!='train':raise ValueError('Reader training artifact contains a held-out group')
        if not engineering_only and row.get('benchmark_dedup_status')!='cleared':
            raise ValueError('Benchmark clearance required outside engineering runs')
        if row.get('positive_label_basis')!='native_program_with_teacher_source_checks':
            raise ValueError('Unrecognized positive supervision')
        if not row['boxes'] or not row['context_boxes']:raise ValueError('Missing source regions')
        if len({n['value'] for n in row['negatives']})!=len(row['negatives']):raise ValueError('Duplicate negatives')
        for negative in row['negatives']:
            group=EXCLUSIVE_GROUPS.get(negative.get('exclusive_group'),())
            oid=negative.get('object_id')
            native=row['native_program_objects'].get(oid,{})
            positive=canonical_class(row['answer']);wrong=canonical_class(negative['value'])
            if (row['query']['predicate']!='category' or negative.get('policy')!=POLICY
                or row['native_evidence'].get('field')!='name'
                or row['native_evidence'].get('object_ids')!=[oid]
                or positive not in group or wrong not in group or wrong==positive
                or canonical_class(native.get('name',''))!=positive
                or negative.get('positive_native_name')!=native.get('name')
                or negative.get('label_basis')!='native_single_object_category_and_explicit_exclusivity'):
                raise ValueError('Negative lacks a compatible native exclusivity record')
        rows.append(row)
    if not rows:raise ValueError('No reader objectives')
    return rows


def backward_objectives(backend,row,identity_weight=0.2,negative_weight=1.0):
    """Accumulate gradients one view at a time to bound activation memory.

    Positive and identity losses use summed sequence log probabilities, including
    EOS. Negative terms use whole-string probabilities, never exp(logp / length).
    The mean of eligible negative strings is weighted independently.
    """
    import torch
    from efa.latent import unlikelihood
    target=[Box(*b) for b in row['boxes']];context=[Box(*b) for b in row['context_boxes']]
    if any(not b.valid(backend.image.width,backend.image.height) for b in target+context):
        raise ValueError('Source geometry outside the original image')
    prompt=read_prompt(row['query'],row['boxes'])
    result={};total=0.0
    def backward(loss,name,weight=1.0):
        nonlocal total
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite '+name+' objective')
        (weight*loss).backward()
        result[name]=float(loss.detach());total+=weight*result[name]
    logp,n=backend.sequence_logp(prompt,target,row['answer'])
    backward(-logp,'read');result['read_tokens']=n
    if identity_weight:
        logp,n=backend.sequence_logp(binding_prompt(row['context_boxes']),context,row['identity'])
        backward(-logp,'identity',identity_weight);result['identity_tokens']=n
    negative_losses=[]
    if negative_weight:
        for index,negative in enumerate(row['negatives']):
            logp,_=backend.sequence_logp(prompt,target,negative['value'])
            name='negative_'+str(index)
            backward(unlikelihood(logp),name,negative_weight/len(row['negatives']))
            negative_losses.append(result.pop(name))
    result.update(negative=sum(negative_losses)/len(negative_losses) if negative_losses else 0.0,
                  negative_count=len(negative_losses),total=total)
    return result
