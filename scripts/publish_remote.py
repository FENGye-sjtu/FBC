"""Project dataset upload with a transient token received through SSH stdin."""
import json
import os
from pathlib import Path,PurePosixPath
import sys


def main():
    payload=json.load(sys.stdin)
    root=Path('/data_cinema/zqc/VBC')
    source=Path(payload['source']).resolve()
    if not source.is_relative_to(root) or source==root:
        raise ValueError('Upload source outside project')
    destination=PurePosixPath(payload['destination'])
    if destination.is_absolute() or '..' in destination.parts or not str(destination).startswith('efa/'):
        raise ValueError('Use a versioned efa/ dataset subfolder')
    os.environ['HF_HOME']=str(root/'cache/huggingface')
    os.environ['HF_ENDPOINT']='https://huggingface.co'
    os.environ['TMPDIR']=str(root/'tmp')
    os.environ['XDG_CACHE_HOME']=str(root/'cache')
    from huggingface_hub import HfApi
    api=HfApi(endpoint='https://huggingface.co',token=payload.pop('token'))
    info=api.dataset_info('QC1/FBC')
    if not info.private:
        raise ValueError('Expected private research dataset')
    manifest=json.loads((source/'manifest.json').read_text())
    if not manifest.get('construction_complete',True):
        raise ValueError('Publish an immutable completed snapshot, not a live writer directory')
    result=api.upload_folder(repo_id='QC1/FBC',repo_type='dataset',revision='main',
        parent_commit=info.sha,folder_path=str(source),path_in_repo=str(destination),
        allow_patterns=['*.json','*.jsonl','*.md','LICENSE.txt','images/*.png','images/*.jpg','images/*.jpeg','images/*.webp'],
        commit_message=f'Add {destination}: {manifest["count"]} region records')
    print(json.dumps(dict(commit=result.oid,manifest=manifest)),flush=True)


if __name__=='__main__':
    main()
