"""Publish a versioned data subfolder without replacing other machines' data."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub.utils import disable_progress_bars, logging as hub_logging


def main():
    disable_progress_bars()
    # Retry logs may contain signed object-storage URLs. Keep upload credentials
    # out of console output while retaining the terminal success/failure result.
    hub_logging.set_verbosity_error()
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--version",required=True)
    a=p.parse_args()
    if not a.version.replace('_','').replace('-','').isalnum():
        raise ValueError("Use an alphanumeric version name")
    root=Path(a.source)
    manifest=json.loads((root/'manifest.json').read_text())
    api=HfApi(endpoint='https://huggingface.co')
    info=api.dataset_info('QC1/FBC',revision='main')
    if not info.private:
        raise RuntimeError('Expected the existing private research repository')
    result=api.upload_folder(repo_id='QC1/FBC',repo_type='dataset',revision='main',
        parent_commit=info.sha,folder_path=str(root),path_in_repo='efa/'+a.version,
        allow_patterns=['*.json','*.jsonl','*.md','LICENSE.txt','images/*.png','images/*.jpg','images/*.jpeg','images/*.webp',
                        'inspection/*.png','inspection/*.jpg','inspection/*.jpeg','inspection/*.webp','inspection/*.py',
                        'inspection/**/*.png','inspection/**/*.jpg','inspection/**/*.jpeg','inspection/**/*.webp',
                        'inspection/**/*.py',
                        'provenance/*.py'],
        commit_message=f'Add EFA {a.version}: {manifest["count"]} region records')
    print(json.dumps(dict(commit=result.oid,url=result.commit_url,version=a.version,
                         count=manifest['count'],kind=manifest.get('kind'))))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        raise SystemExit('Data publication failed: '+type(error).__name__) from None
