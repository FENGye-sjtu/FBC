import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from efa.reader_data import load_training_sources, set_training_image, shuffled_reads


class ReaderDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def episode(self, identifier, status='teacher_reviewed_episode', split='train', digest=None):
        atom = dict(kind='visual', subject='object beside sign', predicate='category', value='cat',
                    native_evidence=dict(object_ids=['5'], field='name'),
                    region=dict(boxes=[[0, 0, 4, 4]], insufficient=False,
                                label_basis='native_box'))
        return dict(id=identifier, split=split, image='image.png',
                    image_sha256=digest or hashlib.sha256(identifier.encode()).hexdigest(),
                    annotation_status=status, label_basis='original_source_label',
                    benchmark_dedup_status='pending', source_repo='example/source', source_revision='pinned',
                    steps=[dict(id=1, atoms=[atom]),
                           dict(id=2, atoms=[dict(kind='premise', value='yes')])])

    def source(self, name, rows, filename='teacher_reviewed_episodes.jsonl', raw=None):
        root = self.root / name
        root.mkdir()
        (root / filename).write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
        if raw is not None:
            (root / 'raw_tasks.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in raw), encoding='utf-8')
        return root

    def test_mix_preserves_tiers_and_excludes_rejected_heldout_and_premises(self):
        approved = self.episode('approved')
        rejected = self.episode('rejected', status='teacher_review_rejected')
        heldout = self.episode('dev', split='dev')
        insufficient = self.episode('insufficient')
        insufficient['steps'][0]['atoms'][0]['region']['insufficient'] = True
        first = self.source('gqa', [approved, rejected, heldout, insufficient])
        native = self.episode('count', status='native_program_episode')
        second = self.source('clevr', [native], 'native_episodes.jsonl')
        rows, manifest = load_training_sources(first, 'teacher_reviewed_episodes.jsonl',
            [(second, 'native_episodes.jsonl')], engineering_only=True)
        self.assertEqual([row['episode_id'] for row in rows], ['approved', 'count'])
        self.assertEqual(rows[1]['annotation_status'], 'native_program_episode')
        self.assertEqual(rows[0]['native_evidence'], approved['steps'][0]['atoms'][0]['native_evidence'])
        self.assertEqual(rows[1]['image_sha256'], native['image_sha256'])
        self.assertEqual(rows[1]['image_path'], str(second / 'image.png'))
        self.assertEqual(rows[0]['source_revision'], 'pinned')
        self.assertEqual(rows[0]['label_basis'], 'original_source_label')
        self.assertEqual(manifest['examples'], 2)
        self.assertEqual(manifest['sources'][0]['excluded_episodes'],
                         dict(annotation_tier_mismatch=1, heldout_episodes=1, no_eligible_visual_atoms=1))
        self.assertEqual(manifest['sources'][1]['sha256'],
                         hashlib.sha256((second / 'native_episodes.jsonl').read_bytes()).hexdigest())
        self.assertEqual(manifest['sources'][1]['eligible_annotation_status_counts'],
                         dict(native_program_episode=1))

    def test_formal_training_requires_cleared_reviewed_tier(self):
        reviewed = self.episode('reviewed', status='reviewed_episode')
        reviewed['benchmark_dedup_status'] = 'cleared'
        pending = self.episode('pending', status='reviewed_episode')
        first = self.source('reviewed', [reviewed, pending], 'reviewed_episodes.jsonl')
        native = self.episode('native', status='native_program_episode')
        native['benchmark_dedup_status'] = 'cleared'
        second = self.source('native', [native], 'native_episodes.jsonl')
        rows, manifest = load_training_sources(first, 'reviewed_episodes.jsonl',
                                               [(second, 'native_episodes.jsonl')])
        self.assertEqual([row['episode_id'] for row in rows], ['reviewed'])
        self.assertEqual(manifest['sources'][0]['excluded_episodes'], dict(benchmark_not_cleared=1))
        self.assertEqual(manifest['sources'][1]['excluded_episodes'], dict(engineering_annotation_tier=1))

    def test_raw_primary_still_uses_neutral_property_extraction(self):
        raw = dict(id='raw', split='train', image='image.png', image_sha256='a'*64,
                   question='What color is the bag?', answer='red', bbox=[0, 0, 4, 4],
                   benchmark_dedup_status='cleared')
        first = self.source('raw', [raw], 'raw_tasks.jsonl')
        rows, _ = load_training_sources(first)
        self.assertEqual(rows[0]['query'], dict(subject='the bag', predicate='color'))
        self.assertEqual(rows[0]['answer'], 'red')
        with self.assertRaisesRegex(ValueError, 'explicit episode annotation tier'):
            load_training_sources(first, episode_sources=[(first, 'raw_tasks.jsonl')])

    def test_unselected_heldout_in_raw_registry_still_blocks_leakage(self):
        first = self.source('gqa', [self.episode('a')])
        native = self.episode('b', status='native_program_episode')
        hidden_heldout = self.episode('heldout', split='calibration', digest=self.episode('a')['image_sha256'])
        second = self.source('native', [native], 'native_episodes.jsonl', raw=[native, hidden_heldout])
        with self.assertRaisesRegex(ValueError, 'Train/held-out image collision'):
            load_training_sources(first, 'teacher_reviewed_episodes.jsonl',
                                   [(second, 'native_episodes.jsonl')], True)

    def test_duplicate_ids_and_inconsistent_split_registry_are_rejected(self):
        episode = self.episode('same')
        first = self.source('one', [episode])
        second = self.source('two', [episode])
        with self.assertRaisesRegex(ValueError, 'Duplicate episode IDs across sources'):
            load_training_sources(first, 'teacher_reviewed_episodes.jsonl',
                                   [(second, 'teacher_reviewed_episodes.jsonl')], True)
        changed = copy.deepcopy(episode)
        changed['split'] = 'dev'
        third = self.source('three', [episode], raw=[changed])
        with self.assertRaisesRegex(ValueError, 'Split or image mismatch'):
            load_training_sources(third, 'teacher_reviewed_episodes.jsonl', engineering_only=True)
        fourth = self.source('four', [episode, episode])
        with self.assertRaisesRegex(ValueError, 'Duplicate episode ID'):
            load_training_sources(fourth, 'teacher_reviewed_episodes.jsonl', engineering_only=True)

    def test_all_source_rows_are_shuffled_per_epoch_reproducibly(self):
        rows = [dict(id=index, source_index=index//4) for index in range(12)]
        order = list(shuffled_reads(rows, 24))
        self.assertEqual(order, list(shuffled_reads(rows, 24)))
        for start in (0, 12):
            epoch = order[start:start+12]
            self.assertEqual(sorted(row['id'] for row in epoch), list(range(12)))
            self.assertNotEqual([row['source_index'] for row in epoch], [row['source_index'] for row in rows])
        self.assertNotEqual(order[:12], order[12:])
        self.assertNotEqual(order, list(shuffled_reads(rows, 24, seed=91)))

    def test_training_checks_pixels_actually_loaded_not_only_manifest(self):
        class ImageBackend:
            def set_image(self, path):
                with Image.open(path) as source:
                    image = source.convert('RGB')
                    self.image_digest = hashlib.sha256(image.tobytes()+str(image.size).encode()).hexdigest()
        path = self.root / 'image.png'
        image = Image.new('RGB', (4, 4), 'red')
        image.save(path)
        row = dict(id='a', image_path=str(path),
                   image_sha256=hashlib.sha256(image.tobytes()+str(image.size).encode()).hexdigest())
        backend = ImageBackend()
        set_training_image(backend, row)
        Image.new('RGB', (4, 4), 'blue').save(path)
        with self.assertRaisesRegex(ValueError, 'Image provenance mismatch'):
            set_training_image(backend, row)


if __name__ == '__main__':
    unittest.main()
