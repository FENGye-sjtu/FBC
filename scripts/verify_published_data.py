"""Download one published dataset version at an exact commit and compare bytes."""
import argparse
import json
from pathlib import Path
import tempfile

from huggingface_hub import HfApi, get_token, snapshot_download
from huggingface_hub.utils import disable_progress_bars, logging as hub_logging


def main():
    disable_progress_bars()
    hub_logging.set_verbosity_error()
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--version',required=True)
    parser.add_argument('--revision',required=True)
    args=parser.parse_args()
    source=Path(args.source).resolve()
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    token=get_token()
    if not token:raise ValueError('Local Hugging Face login is required')
    prefix='efa/'+args.version.strip('/')+'/'
    expected={prefix+'manifest.json',*(prefix+name for name in manifest['files'])}
    api=HfApi(endpoint='https://huggingface.co',token=token)
    info=api.dataset_info('QC1/FBC',revision=args.revision)
    remote={name for name in api.list_repo_files('QC1/FBC',repo_type='dataset',revision=args.revision)
            if name.startswith(prefix)}
    if remote!=expected:
        raise RuntimeError('Published file set mismatch: '+json.dumps({
            'missing':sorted(expected-remote),'unexpected':sorted(remote-expected)}))
    # Hugging Face may briefly retain its local cache marker on Windows after
    # all downloads close; cleanup failure must not replace the byte verdict.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        downloaded=Path(snapshot_download('QC1/FBC',repo_type='dataset',revision=args.revision,
            allow_patterns=[prefix+'*'],local_dir=directory,token=token,max_workers=4))/prefix
        mismatches=[]
        for path in sorted(source.rglob('*')):
            if path.is_file():
                relative=path.relative_to(source)
                if (downloaded/relative).read_bytes()!=path.read_bytes():
                    mismatches.append(relative.as_posix())
    if mismatches:raise RuntimeError('Downloaded byte mismatches: '+json.dumps(mismatches))
    print(json.dumps(dict(repo=info.id,revision=info.sha,version=args.version,
        files=len(expected),byte_mismatches=0,private=info.private)),flush=True)


if __name__=='__main__':main()
