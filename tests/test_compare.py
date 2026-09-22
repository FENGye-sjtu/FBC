import json
from pathlib import Path
import tempfile
import unittest

from trainingfree.compare import compare, mcnemar_exact, wilson_interval


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
            self.assertTrue(result['reference_blind_extraction'])

    def test_confidence_and_paired_statistics(self):
        low, high = wilson_interval(50, 100)
        self.assertLess(low, .5)
        self.assertGreater(high, .5)
        self.assertEqual(mcnemar_exact(0, 0), 1.)
        self.assertAlmostEqual(mcnemar_exact(3, 0), .25)


if __name__ == '__main__':
    unittest.main()
