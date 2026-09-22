import json
import tempfile
import unittest
from pathlib import Path

from trainingfree.eval_protocols import materialize_ags, materialize_vlmeval_cases


class EvalProtocolTests(unittest.TestCase):
    def test_materialize_ags_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, raw, output = root/'cases.jsonl', root/'raw.jsonl', root/'ags.jsonl'
            cases.write_text(json.dumps({'id': 1, 'question': 'q', 'reference': 'A'})+'\n')
            raw.write_text(json.dumps({'id': 1, 'raw': 'reason\nA', 'generated_tokens': 2})+'\n')
            result = materialize_ags(raw, cases, output)
            row = json.loads(output.read_text())
            self.assertEqual(result['n'], 1)
            self.assertEqual(row['model_answer'], 'reason\nA')
            self.assertEqual(row['answer'], 'A')

    def test_materialize_uses_official_prompt_and_mapped_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tsv = root/'data.tsv'
            tsv.write_text('index\tquestion\tA\tB\tanswer\n7\tWhat?\tone\ttwo\tB\n')
            mapping = root/'mapping.json'
            mapping.write_text(json.dumps([{'vlm_index': '7', 'xai_id': 'old'}]))
            originals = root/'original.jsonl'
            originals.write_text(json.dumps({'id': 'old', 'image': 'images/a.jpg'})+'\n')
            output = root/'cases.jsonl'

            result = materialize_vlmeval_cases(tsv, mapping, originals, output)

            case = json.loads(output.read_text())
            self.assertEqual(result['n'], 1)
            self.assertEqual(case['image'], 'images/a.jpg')
            self.assertEqual(case['reference'], 'B')
            self.assertEqual(case['source_xai_id'], 'old')
            self.assertEqual(case['question'],
                             'Question: What?\nOptions:\nA. one\nB. two\n'
                             'Please select the correct answer from the options above.  \n')


if __name__ == '__main__':
    unittest.main()
