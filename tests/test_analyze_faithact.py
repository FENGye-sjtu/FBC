import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from trainingfree.analyze_faithact import analyze, main
from trainingfree.answers import PARSER_VERSION


QUESTION = 'Q\nA. one\nB. two'
COT = 'Evidence:\n\nA visible object.\n\n\n\nFINAL ANSWER: A'


class AnalyzeFaithActTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cases = [{'id': str(i), 'question': QUESTION, 'reference': 'B'} for i in range(3)]
        self.baseline = [{'id': '0', 'raw': COT, 'elapsed_seconds': 2.},
                         {'id': '1', 'raw': 'B', 'elapsed_seconds': 3.}]
        boxes = [[0, 0, 10, 10]]
        self.rows = [
            {'id': '0', 'raw': 'B', 'original_cot': COT, 'evidence': [
                {'noun': 'low', 'probability': .39, 'boxes': boxes},
                {'noun': 'edge', 'probability': .4, 'boxes': boxes},
                {'noun': 'maybe', 'probability': .6, 'boxes': []},
                {'noun': 'high', 'probability': .61, 'boxes': boxes}],
             'extraction_errors': [{'raw': 'not a list', 'error': 'Expected list'}],
             'elapsed_seconds': 7., 'incremental_elapsed_seconds': 5.},
            {'id': '1', 'raw': 'A', 'original_cot': 'B', 'evidence': [], 'extraction_errors': [],
             'elapsed_seconds': 4., 'incremental_elapsed_seconds': 1.},
        ]

    def write_inputs(self):
        paths = {}
        for role, rows in [('raw', self.rows), ('baseline', self.baseline), ('cases', self.cases)]:
            path = self.root / (role + '.jsonl')
            path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')
            paths[role] = path
        return paths

    def test_pairing_bbox_thresholds_calls_and_partial_scope(self):
        report = analyze(**self.write_inputs())
        self.assertEqual(report['parser_version'], PARSER_VERSION)
        self.assertEqual(report['n'], 2)
        self.assertTrue(report['is_partial'])
        self.assertFalse(report['is_partial_baseline_subset'])
        self.assertEqual(report['original_cot_check']['checked_n'], 2)
        self.assertEqual(report['paired_groups']['all']['wins'], ['0'])
        self.assertEqual(report['paired_groups']['all']['losses'], ['1'])
        self.assertEqual(report['paired_groups']['tool_records']['n'], 1)
        self.assertEqual(report['paired_groups']['no_tool_records']['n'], 1)
        self.assertEqual(report['paired_groups']['boxes_in_evidence_prompt']['wins'], ['0'])
        coverage = report['coverage']
        self.assertEqual(coverage['noun_extraction_calls'], 2)  # Includes empty interior paragraph.
        self.assertEqual(coverage['noun_inspections'], 4)
        self.assertEqual(coverage['detected_box_count'], 3)
        self.assertEqual(coverage['shown_box_count'], 2)  # .39 suppressed; .4 included.
        self.assertEqual(coverage['noun_probability_bands'],
                         {'below_0.4': 1, 'maybe_0.4_to_0.6': 2, 'exists_gt_0.6': 1})
        self.assertEqual(report['noun_extraction']['error_count'], 1)
        self.assertEqual(report['noun_extraction']['cases_with_errors'], 1)
        self.assertEqual(report['paired_groups']['all']['seconds']['faithact_incremental']['mean'], 3.)

    def test_rejects_duplicate_ids_in_any_source(self):
        for role in ('raw', 'baseline', 'cases'):
            with self.subTest(role=role):
                paths = self.write_inputs()
                path = paths[role]
                text = path.read_text(encoding='utf-8')
                path.write_text(text + text.splitlines()[0] + '\n', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Duplicate ID'):
                    analyze(**paths)

    def test_exact_initial_cot_mismatch_and_missing_pairs_are_errors(self):
        self.rows[0]['original_cot'] = COT + ' '
        with self.assertRaisesRegex(ValueError, 'does not exactly match'):
            analyze(**self.write_inputs())
        self.rows[0]['original_cot'] = COT
        self.rows[0]['id'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'missing from baseline'):
            analyze(**self.write_inputs())

    def test_empty_raw_is_not_replaced_by_stored_correct_answer(self):
        self.rows[0].update(raw='', answer='B', correct=True)
        self.rows[1].pop('incremental_elapsed_seconds')
        report = analyze(**self.write_inputs())
        self.assertEqual(report['paired_groups']['all']['right_correct'], 0)
        self.assertEqual(report['paired_groups']['all']['faithact_unresolved_mcq'], 1)
        self.assertEqual(report['paired_groups']['all']['seconds']['faithact_incremental']['recorded_n'], 1)
        self.assertEqual(report['paired_groups']['all']['seconds']['faithact_incremental']['mean'], 5.)

    def test_cli_preserves_inputs_and_does_not_overwrite_them(self):
        paths = self.write_inputs()
        before = {role: path.read_bytes() for role, path in paths.items()}
        output = self.root / 'report.json'
        args = [item for role, path in paths.items() for item in ('--' + role, str(path))]
        with contextlib.redirect_stdout(io.StringIO()):
            main(args + ['--output', str(output)])
        self.assertEqual(json.loads(output.read_text(encoding='utf-8'))['n'], 2)
        self.assertEqual({role: path.read_bytes() for role, path in paths.items()}, before)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(args + ['--output', str(paths['raw'])])
        self.assertEqual(paths['raw'].read_bytes(), before['raw'])


if __name__ == '__main__':
    unittest.main()
