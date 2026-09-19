"""Explicit annotation tiers and image provenance for mixed reader alignment."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from efa.native import property_atom


EPISODE_TIERS = {
    'native_episodes.jsonl': 'native_program_episode',
    'teacher_reviewed_episodes.jsonl': 'teacher_reviewed_episode',
    'reviewed_episodes.jsonl': 'reviewed_episode',
}


def _read_file(path):
    payload = path.read_bytes()
    rows = [json.loads(line) for line in payload.decode('utf-8').splitlines() if line.strip()]
    return rows, hashlib.sha256(payload).hexdigest()


def _index(rows, path):
    result = {}
    for row in rows:
        if row['id'] in result:
            raise ValueError(f"Duplicate episode ID {row['id']} in {path}")
        digest = row.get('image_sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f"Missing or invalid image SHA256 for {row['id']} in {path}")
        result[row['id']] = (row['split'], digest)
    return result


def load_training_sources(root, episode_file=None, episode_sources=(), engineering_only=False):
    """Return training visual reads and a manifest without upgrading annotation status.

    The primary source retains the historical raw_tasks fallback. Every additional
    source must name its episode tier explicitly. Full requested files (plus raw
    split registries where present) participate in image/ID collision checks;
    rejected and held-out records never become alignment examples.
    """
    if episode_file is not None and episode_file not in EPISODE_TIERS:
        raise ValueError('Unknown episode annotation tier')
    sources = [(root, episode_file)]
    for extra_root, extra_file in episode_sources:
        if extra_file not in EPISODE_TIERS:
            raise ValueError('Additional sources require an explicit episode annotation tier')
        sources.append((extra_root, extra_file))
    reads, components = [], []
    train_images, heldout_images, episode_ids, read_ids = set(), set(), set(), set()
    for source_index, (source_root, filename) in enumerate(sources):
        source_root = Path(source_root).resolve()
        path = source_root / (filename or 'raw_tasks.jsonl')
        rows, source_sha = _read_file(path)
        identities = _index(rows, path)
        split_registry = None
        raw_path = source_root / 'raw_tasks.jsonl'
        if filename and raw_path.exists():
            raw_rows, raw_sha = _read_file(raw_path)
            raw_index = _index(raw_rows, raw_path)
            for key in identities.keys() & raw_index.keys():
                if identities[key] != raw_index[key]:
                    raise ValueError(f'Split or image mismatch between annotation tiers for {key}')
            identities = dict(raw_index, **identities)
            split_registry = dict(file=str(raw_path), sha256=raw_sha, records=len(raw_rows))
        duplicates = episode_ids & identities.keys()
        if duplicates:
            raise ValueError(f'Duplicate episode IDs across sources: {sorted(duplicates)[:5]}')
        episode_ids.update(identities)
        for split, digest in identities.values():
            (train_images if split == 'train' else heldout_images).add(digest)
        if train_images & heldout_images:
            raise ValueError('Train/held-out image collision across sources or annotation tiers')
        expected = EPISODE_TIERS.get(filename)
        selected_episodes, selected_images, eligible_statuses = set(), set(), Counter()
        exclusions = Counter()
        before = len(reads)
        for row in rows:
            if row['split'] != 'train':
                exclusions['heldout_episodes'] += 1
                continue
            if not engineering_only and row.get('benchmark_dedup_status') != 'cleared':
                exclusions['benchmark_not_cleared'] += 1
                continue
            if filename:
                if row.get('annotation_status') != expected:
                    exclusions['annotation_tier_mismatch'] += 1
                    continue
                if expected != 'reviewed_episode' and not engineering_only:
                    exclusions['engineering_annotation_tier'] += 1
                    continue
                atoms = [(f"{row['id']}-s{step['id']}-a{index}", atom)
                         for step in row['steps'] for index, atom in enumerate(step['atoms'])
                         if atom['kind'] == 'visual' and atom['region'].get('insufficient') is False]
            else:
                atom = property_atom(row['question'], row['answer']) if row.get('bbox') else None
                atoms = [(row['id'], dict(atom, region=dict(boxes=[row['bbox']])))] if atom else []
            for identifier, atom in atoms:
                if identifier in read_ids:
                    raise ValueError(f'Duplicate visual atom ID: {identifier}')
                read_ids.add(identifier)
                reads.append(dict(
                    id=identifier, episode_id=row['id'], image=row['image'],
                    image_path=str(source_root / row['image']), image_sha256=row['image_sha256'],
                    split='train', query={key: atom[key] for key in ('subject', 'predicate')},
                    answer=atom['value'], boxes=atom['region']['boxes'],
                    annotation_status=row.get('annotation_status'), label_basis=row.get('label_basis'),
                    region_label_basis=atom['region'].get('label_basis'),
                    benchmark_dedup_status=row.get('benchmark_dedup_status'),
                    native_evidence=atom.get('native_evidence'), review=row.get('review'),
                    source_repo=row.get('source_repo'), source_revision=row.get('source_revision'),
                    source_index=source_index, source_file=str(path), source_sha256=source_sha))
                selected_episodes.add(row['id'])
                selected_images.add(row['image_sha256'])
                eligible_statuses[row.get('annotation_status', 'unspecified_raw_annotation')] += 1
            if not atoms:
                exclusions['no_eligible_visual_atoms'] += 1
        components.append(dict(
            root=str(source_root), file=str(path), sha256=source_sha,
            episode_file=filename, requested_annotation_status=expected,
            records=len(rows), split_counts=dict(Counter(row['split'] for row in rows)),
            annotation_status_counts=dict(Counter(row.get('annotation_status', 'unspecified_raw_annotation') for row in rows)),
            eligible_episodes=len(selected_episodes), eligible_images=len(selected_images),
            eligible_visual_atoms=len(reads)-before, eligible_annotation_status_counts=dict(eligible_statuses),
            excluded_episodes=dict(exclusions), split_registry=split_registry))
    manifest = dict(
        sources=components, examples=len(reads), episodes=len({row['episode_id'] for row in reads}),
        images=len({row['image_sha256'] for row in reads}), engineering_only=engineering_only,
        source_images_cross_split_checked=True, episode_ids_unique=True,
        objectives=['region_answer_ce'], annotation_statuses_preserved=True)
    return reads, manifest


def shuffled_reads(rows, steps, seed=2027):
    """Mix all source atoms each epoch with an isolated, reproducible RNG."""
    if not rows or steps < 1:
        raise ValueError('No eligible examples or invalid training budget')
    rng = random.Random(seed)
    order = list(range(len(rows)))
    for step in range(steps):
        if step % len(rows) == 0:
            rng.shuffle(order)
        yield rows[order[step % len(rows)]]


def set_training_image(backend, row):
    """Validate the decoded RGB image actually used in this optimizer update."""
    backend.set_image(row['image_path'])
    if backend.image_digest != row['image_sha256']:
        raise ValueError(f"Image provenance mismatch for {row['id']}")
