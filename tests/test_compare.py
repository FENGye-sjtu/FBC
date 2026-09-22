import json
import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest

from trainingfree.compare import compare, mcnemar_exact, wilson_interval
from trainingfree.answers import PARSER_VERSION, extract_answer


class CompareTests(unittest.TestCase):
    def test_reference_blind_pairing_across_run_folders(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = root/'cases.jsonl'
            left = root/'left.jsonl'
            right = root/'right.jsonl'
            dataset.write_text(json.dumps({'id':'0','question':'Q\nA. one\nB. two',
                                           'reference':'B'})+'\n', encoding='utf-8')
            left.write_text(json.dumps({'id':'0','raw':'FINAL ANSWER: A','correct':False,
                                        'status':'cot','elapsed_seconds':1})+'\n', encoding='utf-8')
            right.write_text(json.dumps({'id':'0','raw':'B. two','correct':True,
                                         'status':'verified_by_model','elapsed_seconds':2})+'\n', encoding='utf-8')
            result = compare(dataset, [('base',left),('method',right)])
            self.assertEqual(result['methods']['method']['correct'], 1)
            self.assertEqual(result['paired']['method_vs_base']['wins'], ['0'])
            self.assertEqual(result['paired']['method_vs_base']['left_correct'], 0)
            self.assertEqual(result['paired']['method_vs_base']['right_correct'], 1)
            self.assertFalse(result['paired']['method_vs_base']['input_budget_matched'])
            self.assertEqual(result['paired']['method_vs_base']['input_budget_differences']['model']['reason'],
                             'missing_manifest_value')
            self.assertTrue(result['reference_blind_extraction'])
            self.assertEqual(result['parser_version'],PARSER_VERSION)
            self.assertEqual(result['parser_sha256'], hashlib.sha256(
                Path(inspect.getsourcefile(extract_answer)).read_bytes()).hexdigest())

    def test_matched_subset_and_faithact_evidence_subgroups(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = root/'cases.jsonl'
            dataset.write_text('\n'.join(json.dumps({
                'id': str(i), 'question': 'Q\nA. one\nB. two', 'reference': 'B'})
                for i in range(3))+'\n', encoding='utf-8')
            manifest = {'model':'qwen', 'quantization':'none', 'max_pixels':1003520,
                        'max_new_tokens':512, 'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest()}
            predictions = []
            data = {
                'base': [{'id':'0', 'raw':'B'}, {'id':'1', 'raw':'A'}, {'id':'2', 'raw':'B'}],
                'faithact': [{'id':'1', 'raw':'B', 'method':'faithact', 'evidence':[{'noun':'car'}]},
                             {'id':'2', 'raw':'A', 'method':'faithact', 'evidence':[]}]}
            for name, rows in data.items():
                target = root/name
                target.mkdir()
                (target/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                path = target/'predictions.jsonl'
                path.write_text('\n'.join(map(json.dumps,rows))+'\n', encoding='utf-8')
                predictions.append((name,path))
            result = compare(dataset,predictions)
            pair = result['paired']['faithact_vs_base']
            self.assertTrue(pair['input_budget_matched'])
            self.assertEqual(pair['n'],2)
            self.assertEqual((pair['left_correct'],pair['right_correct']),(1,1))
            evidence = pair['evidence_subgroups']['faithact']['with_evidence']
            absent = pair['evidence_subgroups']['faithact']['without_evidence']
            self.assertEqual((evidence['n'],evidence['left_correct'],evidence['right_correct']),(1,0,1))
            self.assertEqual(evidence['wins'],['1'])
            self.assertEqual((absent['n'],absent['left_correct'],absent['right_correct']),(1,1,0))
            self.assertEqual(absent['losses'],['2'])

            changed_manifest = {**manifest, 'quantization':'nf4', 'max_pixels':200704,
                                'max_new_tokens':256, 'model':'other-qwen'}
            (root/'faithact'/'manifest.json').write_text(json.dumps(changed_manifest), encoding='utf-8')
            pair = compare(dataset,predictions)['paired']['faithact_vs_base']
            self.assertFalse(pair['input_budget_matched'])
            self.assertEqual(set(pair['input_budget_differences']),
                             {'model','quantization','max_pixels','max_new_tokens'})
            self.assertEqual(pair['comparison_scope'],'diagnostic_only_unmatched_input_budget')

            # Matching run hashes are insufficient if both refer to a different dataset.
            for name in data:
                (root/name/'manifest.json').write_text(json.dumps({**manifest,'dataset_sha256':'old'}),
                                                       encoding='utf-8')
            pair = compare(dataset,predictions)['paired']['faithact_vs_base']
            self.assertFalse(pair['input_budget_matched'])
            self.assertEqual(pair['input_budget_differences']['dataset_sha256']['reason'],
                             'different_evaluation_dataset')

    def test_confidence_and_paired_statistics(self):
        low, high = wilson_interval(50, 100)
        self.assertLess(low, .5)
        self.assertGreater(high, .5)
        self.assertEqual(mcnemar_exact(0, 0), 1.)
        self.assertAlmostEqual(mcnemar_exact(3, 0), .25)


if __name__ == '__main__':
    unittest.main()
