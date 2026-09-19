"""Visually adjudicated source/value pairs, separate from student repair traces."""
import hashlib
import json
from pathlib import Path

from efa.format import binding_prompt, read_prompt
from efa.regions import Box


LABEL_BASIS = 'codex_model_visual_adjudication_not_human_gold_or_runtime_audit'


def correction_prompt(row):
    if row['kind'] == 'attribute':
        return read_prompt(row['query'], row['boxes'])
    if row['kind'] == 'binding':
        return binding_prompt(row['boxes'])
    raise ValueError('Unsupported corrective objective; insufficient sources are not negatives')


def load_correction_pairs(root, engineering_only=False):
    root = Path(root).resolve()
    path = root / 'correction_pairs.jsonl'
    payload = path.read_bytes()
    rows = [json.loads(line) for line in payload.decode('utf-8').splitlines() if line.strip()]
    seen = set()
    for row in rows:
        if row['id'] in seen:
            raise ValueError('Duplicate correction ID')
        seen.add(row['id'])
        if row.get('split') != 'train' or row.get('label_basis') != LABEL_BASIS:
            raise ValueError('Corrections require explicit train-only visual adjudication')
        if not engineering_only and row.get('benchmark_dedup_status') != 'cleared':
            raise ValueError('Benchmark clearance required outside engineering runs')
        kind = row.get('kind')
        if kind not in ('attribute', 'binding') or row.get('prompt_family') != 'runtime_' + ('read' if kind == 'attribute' else 'binding'):
            raise ValueError('Unrecognized correction prompt/objective')
        if kind == 'attribute' and (not isinstance(row.get('query'), dict) or set(row['query']) != {'subject', 'predicate'}):
            raise ValueError('Attribute query must exclude the incumbent value')
        if kind == 'binding' and row.get('query') is not None:
            raise ValueError('Binding uses a neutral identity prompt, without claimed identity')
        if not all(isinstance(row.get(key), str) and row[key].strip() for key in ('positive', 'negative', 'parent_episode_id')):
            raise ValueError('Missing positive, negative or parent provenance')
        if row['positive'].strip().casefold() == row['negative'].strip().casefold():
            raise ValueError('Identical positive and negative')
        if not row.get('boxes') or any(len(box) != 4 or any(type(x) is not int for x in box) for box in row['boxes']):
            raise ValueError('Missing native-pixel source geometry')
        prompt = correction_prompt(row)
        if row.get('prompt') != prompt or row.get('prompt_sha256') != hashlib.sha256(prompt.encode('utf-8')).hexdigest():
            raise ValueError('Training prompt differs from the individually reviewed runtime prompt')
        image = (root / row['image']).resolve()
        if not image.is_relative_to(root) or not image.is_file():
            raise ValueError('Correction image is missing or outside the published version')
        row['image_path'] = str(image)
        row['objective'] = 'correction_pair'
    if not rows:
        raise ValueError('No reviewed correction pairs')
    manifest = dict(file=str(path), sha256=hashlib.sha256(payload).hexdigest(),
                    examples=len(rows), images=len({row['image_sha256'] for row in rows}),
                    attributes=sum(row['kind'] == 'attribute' for row in rows),
                    bindings=sum(row['kind'] == 'binding' for row in rows),
                    label_basis=LABEL_BASIS, actual_student_repairs=0)
    return rows, manifest


def attach_reviewed_replay(pairs, reads):
    """Keep every reviewed positive read and check correction parents against it."""
    parents = {(row['episode_id'], row['image_sha256']) for row in reads}
    if any(row.get('label_basis') != LABEL_BASIS or row.get('annotation_status') != 'reviewed_episode' for row in reads):
        raise ValueError('This continuation requires individually adjudicated replay')
    if any((row['parent_episode_id'], row['image_sha256']) not in parents for row in pairs):
        raise ValueError('Correction parent is not an accepted training image')
    return [dict(row, objective='positive_replay') for row in reads] + pairs


def backward_codex_objective(backend, row, negative_weight=1.0):
    """Summed positive CE and whole-sequence UL, using identical source/prompt.

    The two forwards accumulate gradients independently to bound GPU memory.
    Source-swap pairs supervise identity only: a shared true color is untouched.
    """
    import torch
    from efa.latent import unlikelihood

    boxes = [Box(*box) for box in row['boxes']]
    if any(not box.valid(backend.image.width, backend.image.height) for box in boxes):
        raise ValueError('Invalid original-image source')
    pair = row['objective'] == 'correction_pair'
    prompt = correction_prompt(row) if pair else read_prompt(row['query'], row['boxes'])
    positive = row['positive'] if pair else row['answer']
    logp, tokens = backend.sequence_logp(prompt, boxes, positive)
    positive_loss = -logp
    if not torch.isfinite(positive_loss):
        raise RuntimeError('Nonfinite positive loss')
    positive_loss.backward()
    result = dict(positive=float(positive_loss.detach()), positive_tokens=tokens,
                  negative=0.0, negative_tokens=0, objective=row['objective'],
                  kind=row.get('kind', 'visual_read'))
    if pair and negative_weight:
        logp, tokens = backend.sequence_logp(prompt, boxes, row['negative'])
        negative_loss = unlikelihood(logp)
        if not torch.isfinite(negative_loss):
            raise RuntimeError('Nonfinite negative loss')
        (negative_weight * negative_loss).backward()
        result.update(negative=float(negative_loss.detach()), negative_tokens=tokens)
    result['total'] = result['positive'] + negative_weight * result['negative']
    return result
