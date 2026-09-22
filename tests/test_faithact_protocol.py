import unittest

from trainingfree.faithact import refinement_prompt
from trainingfree.prompts import cot_prompt


class Template:
    TEMPLATE_EOT = 'released::{0}::evidence::{1}'


class FaithActProtocolTests(unittest.TestCase):
    def test_released_gcot_prompt_is_separate_and_keeps_task(self):
        prompt = cot_prompt('Question: Is the door open?\nA. Yes\nB. No',
                            'faithact_gcot_official')
        self.assertTrue(prompt.startswith('\nYou are a visual reasoning assistant.'))
        self.assertIn('Step 1. Grounding:', prompt)
        self.assertIn('Step 2. Reasoning:', prompt)
        self.assertIn('Step 3. Answer:', prompt)
        self.assertIn('A. Yes\nB. No', prompt)
        self.assertNotIn('Question: Question:', prompt)
        self.assertNotIn('FINAL ANSWER:', prompt)

    def test_released_refinement_remains_byte_compatible(self):
        prompt = refinement_prompt(Template, 'released_code', 'Q', 'OLD', ['car: exists'])
        self.assertEqual(prompt, 'released::Q::evidence::car: exists')
        self.assertNotIn('OLD', prompt)

    def test_paper_literal_cot_is_not_mislabeled_as_gcot(self):
        prompt = cot_prompt('Question: Is the door open?\nA. Yes\nB. No',
                            'faithact_paper_cot_literal')
        self.assertEqual(prompt, 'Question: Is the door open?\nA. Yes\nB. No\nThink step by step.')
        self.assertNotIn('Grounding:', prompt)

    def test_paper_appendix_protocol_includes_original_response(self):
        prompt = refinement_prompt(Template, 'paper_appendix', 'Q', 'ORIGINAL_CHAIN',
                                   ['car: exists'])
        self.assertIn('Question: Q', prompt)
        self.assertIn('Model Response: ORIGINAL_CHAIN', prompt)
        self.assertIn('car: exists', prompt)
        self.assertIn('marked as absent', prompt)

    def test_unknown_protocol_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown FaithAct'):
            refinement_prompt(Template, 'mixed', 'Q', 'R', [])

if __name__ == '__main__':
    unittest.main()
