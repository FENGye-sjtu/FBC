"""Fetch pinned upstream assets and materialize RealWorldQA, within CVPR only."""
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import time
import shutil

ROOT = Path('/data_cinema/zqc/CVPR')
UPSTREAM_SHA = 'f0681f4c8d05fa415ee29f3906faa2bdcdf790ab'
POLL_SHA = 'fc4b00a6c00be51e30cdec55f9b9e2c47afe5081082edf93cbdcfef92b68ec5a'


def checkpoint():
    import requests
    path = ROOT/'models/faithact_poll/best.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == POLL_SHA:
        return str(path)
    url = f'https://media.githubusercontent.com/media/lijunxian111/Faithful-First-RPA/{UPSTREAM_SHA}/train_pope/best.pt'
    part = path.with_suffix('.part')
    with requests.get(url, stream=True, timeout=(30,120)) as response:
        response.raise_for_status()
        digest = hashlib.sha256()
        with part.open('wb') as out:
            for block in response.iter_content(4*1024*1024):
                out.write(block); digest.update(block)
    if digest.hexdigest() != POLL_SHA:
        raise ValueError('FaithAct checkpoint hash mismatch')
    part.replace(path)
    return str(path)


def model(repo, folder):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    revision = api.model_info(repo).sha
    # Reuse an existing official snapshot read-only, after checking every weight
    # against the pinned HF LFS hash. Only symlinks/copies under CVPR are created.
    cached = Path('/data_cinema/zqc/covt_internalization/models/Qwen2.5-VL-7B-Instruct')
    if repo == 'Qwen/Qwen2.5-VL-7B-Instruct' and cached.is_dir():
        info = api.model_info(repo, revision=revision, files_metadata=True)
        weights = [s for s in info.siblings if s.rfilename.endswith('.safetensors')]
        destination = ROOT/'models'/folder
        destination.mkdir(parents=True, exist_ok=True)
        verified = []
        for sibling in weights:
            source = cached/sibling.rfilename
            if not source.is_file() or source.stat().st_size != sibling.size:
                break
            digest = hashlib.sha256()
            with source.open('rb') as stream:
                for chunk in iter(lambda: stream.read(16*1024*1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != sibling.lfs.sha256:
                break
            verified.append((source, sibling.rfilename))
        if len(verified) == len(weights) and weights:
            for source, name in verified:
                target = destination/name
                if not target.exists():
                    target.symlink_to(source)
            for source in cached.iterdir():
                if source.is_file() and source.suffix in ('.json','.txt'):
                    metadata = cached/'.cache/huggingface/download'/(source.name+'.metadata')
                    if not metadata.is_file() or metadata.read_text().splitlines()[0] != revision:
                        raise ValueError(f'Cached config revision mismatch: {source.name}')
                    shutil.copy2(source, destination/source.name)
            return {'repo':repo, 'revision':revision, 'path':str(destination),
                    'read_only_weight_source':str(cached), 'all_weight_lfs_hashes_verified':True}
    ignored = ['*.bin','*.msgpack','*.h5','*.onnx']
    if repo == 'openai/clip-vit-large-patch14-336':
        ignored += ['*.safetensors']  # Full encoder weights are inside verified best.pt.
    location = snapshot_download(repo, revision=revision, local_dir=str(ROOT/'models'/folder),
                                 max_workers=2, ignore_patterns=ignored)
    if repo != 'openai/clip-vit-large-patch14-336' and not (list(Path(location).glob('*.safetensors')) or list(Path(location).glob('pytorch_model*.bin'))):
        raise ValueError(f'No model weights downloaded for {repo}')
    return {'repo': repo, 'revision': revision, 'path': location,
            'weights_in_poll_checkpoint': repo == 'openai/clip-vit-large-patch14-336'}


def dataset():
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, snapshot_download
    from PIL import Image
    repo = 'xai-org/RealworldQA'
    revision = HfApi().dataset_info(repo).sha
    source = snapshot_download(repo, repo_type='dataset', revision=revision,
                               local_dir=str(ROOT/'data/raw/realworldqa'), allow_patterns=['*.parquet','README.md'])
    output = ROOT/'data/realworldqa'
    (output/'images').mkdir(parents=True, exist_ok=True)
    rows = []
    for file in sorted(Path(source).rglob('*.parquet')):
        for row in pq.read_table(file).to_pylist():
            index = len(rows)
            image_path = output/'images'/f'{index:04d}.png'
            Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB').save(image_path)
            rows.append({'id': str(index), 'image': str(image_path.relative_to(output)),
                         'question': row['question'], 'reference': row['answer']})
    if len(rows) != 765:
        raise ValueError(f'Expected 765 RealWorldQA samples, got {len(rows)}')
    (output/'cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    return {'repo': repo, 'revision': revision, 'n': len(rows), 'path': str(output)}


def main():
    jobs = {'poll': checkpoint,
            'qwen7b': lambda: model('Qwen/Qwen2.5-VL-7B-Instruct','Qwen2.5-VL-7B-Instruct'),
            'clip': lambda: model('openai/clip-vit-large-patch14-336','clip-vit-large-patch14-336'),
            'dino': lambda: model('IDEA-Research/grounding-dino-base','grounding-dino-base'),
            'realworldqa': dataset}
    status = {}
    ROOT.joinpath('runtime').mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fn): name for name, fn in jobs.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                status[name] = {'state': 'ready', 'result': future.result()}
            except Exception as exc:
                status[name] = {'state': 'failed', 'error': str(exc)}
            status[name]['timestamp'] = time.time()
            (ROOT/'runtime/assets.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
            print(json.dumps({name: status[name]}), flush=True)


if __name__ == '__main__':
    main()
