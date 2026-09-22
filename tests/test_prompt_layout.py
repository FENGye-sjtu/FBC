import hashlib
import unittest

from trainingfree.prompts import cot_prompt, task_question


QUESTION = ('Question: Which object is visible?\nOptions:\nA. Cat\nB. Dog\n'
            'Please select the correct answer from the options above.')


class PromptLayoutTests(unittest.TestCase):
    def test_existing_prompt_variants_remain_byte_identical(self):
        # Captured before adding v5. Running jobs keep their original prompts.
        expected = {
            'reasoned_v2':'bc1d877b4816a2c31533498050f65decdc3cd0d7053eae6dfd144ffee3a8510a',
            'grounded_concise_v3':'58f98080fb6230446507f8557be50f38a4be732f9bada23640bc41f22b58cbcd',
            'preserve_select_v3':'c13009e3201790e1bdc9939fcabb67b8b4f8057d10a49d1a3394c22a099b6e4b',
            'evidence_brief_v4':'9993117d94f7f6593da00a80c15eb71555860f284c27a32a7a97f285bd20cd50',
            'evidence_adaptive_v4':'ba3ff226efd4362bff99996cce03ab631dce7b670f4edb63e9057c78ce7705f1'}
        for style,digest in expected.items():
            with self.subTest(style=style):
                self.assertEqual(hashlib.sha256(cot_prompt(QUESTION,style).encode()).hexdigest(),digest)

    def test_literal_layout_preserves_task_without_a_favored_option(self):
        prompt = cot_prompt(QUESTION,'evidence_layout_v5')
        self.assertTrue(prompt.startswith(task_question(QUESTION)+'\n\n'))
        self.assertIn('A. Cat\nB. Dog',prompt)
        self.assertIn('Copy the blank-line layout below exactly',prompt)
        self.assertTrue(prompt.endswith('Evidence:\n\n1. <visible fact>\n\nFINAL ANSWER: <option>'))
        self.assertNotIn('FINAL ANSWER: A',prompt)
        self.assertNotIn('FINAL ANSWER: B',prompt)

    def test_requested_layout_is_faithact_compatible_without_rewriting_output(self):
        # This checks the requested format, not whether a model obeys it.
        for evidence in ('1. A visible animal has pointed ears.',
                         '1. A visible animal has pointed ears.\n\n2. Its tail is curled.'):
            response = 'Evidence:\n\n'+evidence+'\n\nFINAL ANSWER: A'
            self.assertEqual(response.split('\n\n')[1:-1],evidence.split('\n\n'))

    def test_open_and_binary_answer_formats_are_scoped_to_final_line(self):
        binary = cot_prompt('Question: Is the door open?','evidence_layout_v5')
        self.assertIn('Only the final answer line should contain Yes or No.',binary)
        self.assertTrue(binary.endswith('FINAL ANSWER: <answer>'))
        free = cot_prompt('How many objects are visible?','evidence_layout_v5')
        self.assertIn('Only the final answer line should contain a short word or number.',free)


if __name__ == '__main__':
    unittest.main()
