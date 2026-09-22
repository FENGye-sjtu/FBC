from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from trainingfree.publish_stage import ENDPOINT, REPO, StagePublishError, main, publish_stage


class FakeApi:
    endpoint = ENDPOINT

    def __init__(self, private=True, existing=()):
        self.private, self.existing = private, existing
        self.calls = []

    def repo_info(self, **kwargs):
        self.calls.append(('info', kwargs))
        return SimpleNamespace(private=self.private, sha='a' * 40)

    def list_repo_files(self, **kwargs):
        self.calls.append(('list', kwargs))
        return self.existing

    def create_commit(self, **kwargs):
        self.calls.append(('commit', kwargs))
        return SimpleNamespace(oid='b' * 40)


class PublishStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = b'{"measured": true}\n'
        (self.root / 'results.jsonl').write_bytes(self.data)
        self.manifest()

    def manifest(self, **extra):
        (self.root / 'manifest.json').write_text(json.dumps(
            {'construction_complete': True, **extra}), encoding='utf-8')

    def test_fixed_private_destination_and_atomic_parent(self):
        api = FakeApi()
        result = publish_stage(self.root, '20260922_complete', api)
        self.assertEqual(result['files'], 2)
        self.assertEqual(api.calls[0][1], {'repo_id': REPO, 'repo_type': 'dataset', 'revision': 'main'})
        self.assertEqual(api.calls[1][1]['revision'], 'a' * 40)
        commit = api.calls[2][1]
        self.assertEqual(commit['parent_commit'], 'a' * 40)
        self.assertEqual(commit['revision'], 'main')
        self.assertFalse(commit['create_pr'])
        self.assertEqual({op.path_in_repo for op in commit['operations']}, {
            'trainingfree/milestones/20260922_complete/manifest.json',
            'trainingfree/milestones/20260922_complete/results.jsonl'})
        self.assertNotIn('token', commit)

    def test_whitelist_hashes_and_exact_selection(self):
        (self.root / 'unlisted.txt').write_text('not selected')
        digest = hashlib.sha256(self.data).hexdigest()
        self.manifest(files={'results.jsonl': digest})
        api = FakeApi()
        self.assertEqual(publish_stage(self.root, 'stage', api)['files'], 2)
        (self.root / 'results.jsonl').write_text('changed')
        with self.assertRaisesRegex(StagePublishError, '^ValueError$'):
            publish_stage(self.root, 'stage', FakeApi())

    def test_list_whitelist(self):
        self.manifest(files=[{'path': 'results.jsonl', 'sha256': hashlib.sha256(self.data).hexdigest()}])
        self.assertEqual(publish_stage(self.root, 'stage', FakeApi())['files'], 2)

    def test_incomplete_missing_manifest_and_unsafe_version_make_no_api_calls(self):
        for change in (lambda: self.manifest(construction_complete=False),
                       lambda: (self.root / 'manifest.json').unlink()):
            self.manifest()
            change()
            api = FakeApi()
            with self.assertRaises(StagePublishError):
                publish_stage(self.root, 'stage', api)
            self.assertEqual(api.calls, [])
        self.manifest()
        api = FakeApi()
        with self.assertRaises(StagePublishError):
            publish_stage(self.root, '../stage', api)
        self.assertEqual(api.calls, [])

    def test_public_repo_and_existing_destination_are_rejected(self):
        for api in (FakeApi(private=False), FakeApi(existing=['trainingfree/milestones/stage/old.json'])):
            with self.assertRaises(StagePublishError):
                publish_stage(self.root, 'stage', api)
            self.assertFalse(any(name == 'commit' for name, _ in api.calls))

    def test_forbidden_files_and_path_escape(self):
        for name in ('credentials.json', '.env.txt', 'secret.py', 'image.png'):
            path = self.root / name
            path.write_text('private')
            with self.assertRaises(StagePublishError):
                publish_stage(self.root, 'stage', FakeApi())
            path.unlink()
        self.manifest(files={'../outside.json': 'a' * 64})
        with self.assertRaises(StagePublishError):
            publish_stage(self.root, 'stage', FakeApi())

    def test_error_diagnostics_hide_signed_urls_and_do_not_retry(self):
        class FailingApi(FakeApi):
            def create_commit(self, **kwargs):
                self.calls.append(('commit', kwargs))
                print('https://example.invalid/?secret=signed')
                raise RuntimeError('https://example.invalid/?secret=signed')
        api, output = FailingApi(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            with self.assertRaisesRegex(StagePublishError, '^RuntimeError$'):
                publish_stage(self.root, 'stage', api)
        self.assertEqual(output.getvalue(), '')
        self.assertEqual(sum(name == 'commit' for name, _ in api.calls), 1)
        with patch('trainingfree.publish_stage.publish_stage', side_effect=StagePublishError('RuntimeError')):
            with redirect_stderr(output):
                self.assertEqual(main(['--source', str(self.root), '--version', 'stage']), 1)
        self.assertEqual(json.loads(output.getvalue()), {'error': 'RuntimeError'})


if __name__ == '__main__':
    unittest.main()
