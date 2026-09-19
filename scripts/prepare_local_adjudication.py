"""Stage existing original images for fresh individual annotation, without relabeling."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def prepare(selection_path, output, exclude_roots):
    selection_path, output = Path(selection_path), Path(output)
    if output.exists():
        raise FileExistsError('Choose a fresh candidate directory')
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    selected = selection if isinstance(selection, list) else selection['candidates']
    excluded_hashes = set()
    exclusions = []
    for root in map(Path, exclude_roots):
        source = root / 'reviewed_episodes.jsonl'
        payload = source.read_bytes()
        excluded_hashes.update(json.loads(line)['image_sha256'] for line in payload.decode().splitlines() if line.strip())
        exclusions.append(dict(file=str(source), sha256=hashlib.sha256(payload).hexdigest()))
    indices, prepared, images, seen_ids, seen_hashes = {}, [], {}, set(), set()
    for entry in selected:
        root = Path(entry['source_root']).resolve()
        if root not in indices:
            source = root / 'raw_tasks.jsonl'
            payload = source.read_bytes()
            rows = [json.loads(line) for line in payload.decode('utf-8').splitlines() if line.strip()]
            index = {row['id']: row for row in rows}
            if len(index) != len(rows):
                raise ValueError('Duplicate source IDs')
            indices[root] = (index, hashlib.sha256(payload).hexdigest())
        index, source_sha = indices[root]
        row = dict(index[entry['id']])
        if row['id'] in seen_ids or row['image_sha256'] in seen_hashes | excluded_hashes:
            raise ValueError('Selected image already staged/adjudicated: ' + row['id'])
        path = (root / row['image']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Image source escapes its dataset root')
        content = path.read_bytes()
        with Image.open(path) as im:
            rgb = im.convert('RGB')
        digest = hashlib.sha256(rgb.tobytes() + str(rgb.size).encode()).hexdigest()
        if digest != row['image_sha256'] or rgb.size != (row['width'], row['height']):
            raise ValueError('Image pixels differ from source manifest')
        # Preserve the exact original compressed bytes, split and source labels.
        seen_ids.add(row['id'])
        seen_hashes.add(digest)
        images[row['image']] = content
        row['prior_local_annotation_status'] = row.get('annotation_status')
        row['local_source_provenance'] = dict(root=str(root), file='raw_tasks.jsonl', sha256=source_sha)
        row['selection_record'] = entry
        row['source_trace_reused'] = False
        row['annotation_status'] = 'source_qa_requires_episode_conversion'
        row['steps'] = None
        row['runtime_audit'] = None
        row['image_quality_status'] = 'pending_individual_visual_adjudication'
        prepared.append(row)
    if not prepared:
        raise ValueError('No candidates selected')
    output.mkdir(parents=True)
    for name, content in images.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (output / 'selection.json').write_bytes(selection_bytes)
    (output / 'raw_tasks.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in prepared), encoding='utf-8')
    manifest = dict(kind='local_original_images_pending_codex_adjudication', count=len(prepared),
        source_counts=dict(Counter(row['source_repo'] for row in prepared)),
        split_counts=dict(Counter(row['split'] for row in prepared)),
        approved=0, original_image_bytes_preserved=True, original_splits_preserved=True,
        selection_sha256=hashlib.sha256(selection_bytes).hexdigest(),
        reviewed_image_exclusions=exclusions, benchmark_dedup_status='pending', engineering_only=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--selection', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--exclude-reviewed', action='append', default=[])
    a = p.parse_args()
    print(json.dumps(prepare(a.selection, a.output, a.exclude_reviewed)), flush=True)


if __name__ == '__main__':
    main()
