import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from trainingfree.evaluation_suite import evaluate
from trainingfree.answers import PARSER_VERSION


class EvaluationSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cases = self.root/'input_cases.jsonl'
        self.raw = self.root/'input_raw.jsonl'
        self.output = self.root/'evaluation'
        self.case_rows = [
            {'id':'0','question':'Q\nA. left\nB. right','reference':'B'},
            {'id':'1','question':'Q\nA. left\nB. right','reference':'A'}]
        self.rows = [
            {'id':'0','raw':'A was considered.\n\nFINAL ANSWER: B','answer':'A','correct':False,
             'original_cot':'Evidence:\n\n1. A visible car points right.\n\nFINAL ANSWER: B',
             'evidence':[{'noun':'car'}]},
            {'id':'1','raw':'B','answer':'B','correct':False}]
        self.write(self.cases,self.case_rows)
        self.write(self.raw,self.rows)

    @staticmethod
    def write(path, rows):
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')

    def test_exact_coverage_and_immutable_outputs(self):
        original = self.raw.read_bytes()
        result = evaluate(self.raw,self.cases,self.output)
        self.assertEqual((result['completed_n'],result['expected_n'],result['correct']),(2,2,1))
        self.assertEqual(result['valid_label_n'],2)
        self.assertEqual(result['with_evidence_n'],1)
        self.assertEqual(result['with_interior_cot_n'],1)
        self.assertEqual(result['stored_correctness_changed_n'],1)
        self.assertEqual(result['parser_version'], PARSER_VERSION)
        self.assertEqual(self.raw.read_bytes(),original)
        self.assertEqual((self.output/'raw_predictions.jsonl').read_bytes(),original)
        self.assertEqual(len((self.output/'disagreements.jsonl').read_text().splitlines()),1)

    def test_missing_unknown_duplicate_ids_and_output_collision_rejected(self):
        self.write(self.raw,self.rows[:1])
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            evaluate(self.raw,self.cases,self.output)
        result = evaluate(self.raw,self.cases,self.output,allow_partial=True)
        self.assertTrue(result['is_partial'])
        self.assertEqual(result['missing_ids'],['1'])
        self.write(self.raw,self.rows+[self.rows[0]])
        with self.assertRaisesRegex(ValueError,'Duplicate prediction'):
            evaluate(self.raw,self.cases,self.output)
        self.write(self.raw,[{**self.rows[0],'id':'unknown'}])
        with self.assertRaisesRegex(ValueError,'Unknown'):
            evaluate(self.raw,self.cases,self.output,allow_partial=True)
        self.write(self.raw,self.rows)
        self.write(self.cases,self.case_rows+[self.case_rows[0]])
        with self.assertRaisesRegex(ValueError,'Duplicate dataset'):
            evaluate(self.raw,self.cases,self.output)
        self.write(self.cases,self.case_rows)
        source_in_output = self.root/'raw_predictions.jsonl'
        self.write(source_in_output,self.rows)
        with self.assertRaisesRegex(ValueError,'overwrite'):
            evaluate(source_in_output,self.cases,self.root)

    def test_upstream_raw_and_canonical_are_separate_and_ags_counts_are_actual(self):
        ags_source = self.root/'AGS/extract_ans/getResult_nlp.py'
        ags_source.parent.mkdir(parents=True)
        ags_source.write_text('# evaluator fixture, subprocess is mocked\n', encoding='utf-8')
        matching_source = self.root/'matching_util.py'
        matching_source.write_text('# matching fixture\n', encoding='utf-8')
        def infer(text, choices):
            return 'A' if text.startswith('A') else ('B' if text == 'B' else None)
        completed = subprocess.CompletedProcess([],0,'Total: 2\nCorrect: 0\nACC: 0.000000\n','')
        with patch('trainingfree.evaluation_suite.load_vlmeval_can_infer',
                   new=lambda root: (infer,matching_source)), \
             patch('trainingfree.evaluation_suite.subprocess.run', return_value=completed) as run:
            result = evaluate(self.raw,self.cases,self.output,ags_root=self.root/'AGS',
                              vlmeval_root=self.root/'VLMEvalKit')
        self.assertEqual(result['scorers']['vlmeval_raw']['correct'],0)
        self.assertEqual(result['scorers']['vlmeval_canonical_final']['correct'],1)
        self.assertEqual(result['scorers']['ags_raw']['correct'],0)
        self.assertFalse(result['scorers']['ags_raw']['per_case_available'])
        self.assertEqual(result['correctness_disagreement_n'],1)
        self.assertIn('--input-files',run.call_args.args[0])
        materialized = [json.loads(line) for line in (self.output/'ags_input.jsonl').read_text().splitlines()]
        self.assertEqual(materialized[0]['model_answer'],self.rows[0]['raw'])
        self.assertEqual(materialized[0]['answer'],'B')
        self.assertEqual((self.output/'ags_stdout.txt').read_text(),completed.stdout)

    def test_missing_null_and_empty_raw_answers_and_vlmeval_none(self):
        rows = [
            {'id':'0','answer':'B','status':'verified_by_model'},
            {'id':'1','raw':None,'answer':'B','status':'answered_with_uncertainty'},
            {'id':'2','raw':'','answer':'B','status':'faithact_port'},
            {'id':'3','answer':None},
            {'id':'4','raw':None,'missing':'B'},
            {'id':'5','raw':'FINAL ANSWER: B','answer':'A'}]
        self.write(self.raw,rows)
        self.write(self.cases,[{**self.case_rows[0],'id':str(i)} for i in range(len(rows))])
        matching_source = self.root/'matching_util.py'
        matching_source.write_text('# matching fixture\n', encoding='utf-8')
        seen = []
        def infer(text, choices):
            self.assertIsInstance(text,str)
            seen.append(text)
            return 'B' if text.endswith('B') else None
        with patch('trainingfree.evaluation_suite.load_vlmeval_can_infer',
                   new=lambda root: (infer,matching_source)):
            result = evaluate(self.raw,self.cases,self.output,vlmeval_root=self.root/'VLMEvalKit')
        self.assertEqual(result['correct'],3)
        self.assertEqual(result['valid_label_n'],3)
        self.assertEqual(result['scorers']['vlmeval_raw']['correct'],3)
        details = [json.loads(line) for line in (self.output/'details.jsonl').read_text().splitlines()]
        self.assertEqual([row['answer_source'] for row in details],
                         ['answer','answer','raw','missing','missing','raw'])
        self.assertEqual([row['raw'] for row in details],['B','B','','','','FINAL ANSWER: B'])
        for row in details[2:5]:
            for scorer in ('vlmeval_raw','vlmeval_canonical_final'):
                self.assertIsNone(row['scorers'][scorer]['upstream_answer'])
                self.assertEqual(row['scorers'][scorer]['answer'],'Z')
                self.assertFalse(row['scorers'][scorer]['correct'])
        self.assertEqual(len(seen),12)

    def test_null_extracted_answer_is_passed_to_vlmeval_as_empty_text(self):
        matching_source = self.root/'matching_util.py'
        matching_source.write_text('# matching fixture\n', encoding='utf-8')
        seen = []
        def infer(text, choices):
            self.assertIsInstance(text,str)
            seen.append(text)
            return None
        with patch('trainingfree.evaluation_suite.extract_answer',new=lambda text,question: None), \
             patch('trainingfree.evaluation_suite.load_vlmeval_can_infer',
                   new=lambda root: (infer,matching_source)):
            result = evaluate(self.raw,self.cases,self.output,vlmeval_root=self.root/'VLMEvalKit')
        self.assertEqual(result['correct'],0)
        self.assertEqual(seen[1::2],['',''])


if __name__ == '__main__':
    unittest.main()
