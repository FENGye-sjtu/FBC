"""Publish an already constructed snapshot to a new private HF milestone.

No bundling or source-file copying is performed. Optional manifest ``files`` is
an exact whitelist: {"relative/file.jsonl": "sha256"}, or a list of objects
containing ``path`` and ``sha256``. The root manifest is always included.
Authentication is left to HfApi's cached/environment credentials.
"""
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import io
import json
import logging
from pathlib import Path, PurePosixPath
import re
import sys


REPO = 'QC1/FBC'
ENDPOINT = 'https://huggingface.co'
EXTENSIONS = {'.json', '.jsonl', '.md', '.txt', '.py'}


class StagePublishError(Exception):
    """The message contains an exception type only, never an API response."""


def _safe_name(name):
    path = PurePosixPath(name)
    if not name or '\\' in name or path.is_absolute() or ':' in name or any(
            part in {'.', '..'} for part in name.split('/')):
        raise ValueError()
    if any(part.lower().startswith('.env') or re.search(
            r'(^|[._-])(secrets?|credentials?|tokens?)([._-]|$)', part, re.I)
           for part in path.parts):
        raise ValueError()
    if path.suffix.lower() not in EXTENSIONS:
        raise ValueError()
    return path.as_posix()


def _snapshot(source):
    root = Path(source)
    if root.is_symlink() or not root.is_dir():
        raise ValueError()
    root = root.resolve()
    files = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError()
        if not path.is_file():
            continue
        path.resolve().relative_to(root)
        name = _safe_name(path.relative_to(root).as_posix())
        files[name] = path
    if 'manifest.json' not in files:
        raise ValueError()
    manifest_bytes = files['manifest.json'].read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or manifest.get('construction_complete') is not True:
        raise ValueError()
    whitelist = manifest.get('files')
    if whitelist is not None:
        if isinstance(whitelist, list):
            entries = [(item['path'], item['sha256']) for item in whitelist]
        elif isinstance(whitelist, dict):
            entries = [(name, value['sha256'] if isinstance(value, dict) else value)
                       for name, value in whitelist.items()]
        else:
            raise ValueError()
        wanted = {}
        for name, digest in entries:
            name = _safe_name(name)
            if name in wanted or name not in files or not isinstance(digest, str) or not re.fullmatch(
                    r'[0-9a-fA-F]{64}', digest):
                raise ValueError()
            wanted[name] = digest.lower()
    else:
        wanted = dict.fromkeys(files)
    payloads = {'manifest.json': manifest_bytes}
    for name, expected in wanted.items():
        data = manifest_bytes if name == 'manifest.json' else files[name].read_bytes()
        if expected is not None and hashlib.sha256(data).hexdigest() != expected:
            raise ValueError()
        payloads[name] = data
    return payloads


@contextmanager
def _quiet():
    previous = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        # SDK progress and response diagnostics must not expose signed URLs.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            yield
    finally:
        logging.disable(previous)


def publish_stage(source, version, api=None):
    """Upload once; concurrent branch movement fails without automatic retry."""
    try:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', version or ''):
            raise ValueError()
        payloads = _snapshot(source)
        destination = f'trainingfree/milestones/{version}'
        with _quiet():
            from huggingface_hub import CommitOperationAdd, HfApi
            api = HfApi(endpoint=ENDPOINT) if api is None else api
            if api.endpoint.rstrip('/') != ENDPOINT:
                raise ValueError()
            info = api.repo_info(repo_id=REPO, repo_type='dataset', revision='main')
            if info.private is not True or not re.fullmatch(r'[0-9a-fA-F]{7,64}', info.sha or ''):
                raise ValueError()
            existing = api.list_repo_files(repo_id=REPO, repo_type='dataset', revision=info.sha)
            if any(name == destination or name.startswith(destination + '/') for name in existing):
                raise FileExistsError()
            operations = [CommitOperationAdd(path_in_repo=f'{destination}/{name}', path_or_fileobj=data)
                          for name, data in sorted(payloads.items())]
            commit = api.create_commit(repo_id=REPO, repo_type='dataset', revision='main',
                                       parent_commit=info.sha, create_pr=False, operations=operations,
                                       commit_message=f'Add training-free milestone {version}')
            if not re.fullmatch(r'[0-9a-fA-F]{7,64}', commit.oid or ''):
                raise ValueError()
        return {'repo': REPO, 'revision': 'main', 'destination': destination,
                'commit': commit.oid, 'files': len(payloads)}
    except Exception as exc:
        raise StagePublishError(type(exc).__name__) from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--version', required=True)
    args = parser.parse_args(argv)
    try:
        result = publish_stage(args.source, args.version)
    except StagePublishError as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
