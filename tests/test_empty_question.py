import json
import unittest

from PIL import Image

from trainingfree.core import Step, solve
from trainingfree.online import Online, sanitize_evidence_question
from trainingfree.online_repair import OnlineRepair


QUESTION = ('Question: Which direction does the wheel face?\nOptions:\n'
            'A. Left\nB. Right\nPlease select the correct answer from the options above.')


class EmptyQueryVLM:
    def __init__(self, conditional=False):
        self.proposals = 0
        self.audit_calls = 0
        self.conditional = conditional

    def generate(self, image, prompt, schema=None, **kwargs):
        if 'box' in schema['required']:
            return json.dumps({'box': [10, 10, 60, 60]})
        if 'observation' in schema['required']:
            return json.dumps({'observation': 'The wheel orientation remains ambiguous.',
                               'remaining_question': '', 'evidence_complete': False})
        self.proposals += 1
        deduction = self.conditional and self.proposals > 4
        return json.dumps({
            'claim': ('Given the unresolved cue, option B is a conditional answer.' if deduction else
                      f'The wheel has an ambiguous orientation in observation {self.proposals}.'),
            'kind': 'deduction' if deduction else 'visual',
            'evidence_question': '' if self.proposals % 2 else '  \n ',
            'final_answer': 'B',
        })

    def yes_probability(self, image, prompt):
        self.audit_calls += 1
        return .9 if self.conditional and self.proposals > 4 else .1


class EmptyQuestionTests(unittest.TestCase):
    def test_visual_fallback_is_public_question_without_choices(self):
        original = Step('A tentative visual observation.', evidence_question='  \n ')
        fixed = sanitize_evidence_question(original, QUESTION)
        self.assertEqual(fixed.evidence_question, 'Which direction does the wheel face?')
        self.assertNotIn('Left', fixed.evidence_question)
        self.assertEqual(fixed.verification, 'candidate')
        deduction = sanitize_evidence_question(
            Step('A conditional inference.', kind='deduction', evidence_question=' \n'), QUESTION)
        self.assertEqual(deduction.evidence_question, '')

    def test_reworded_empty_visual_claims_cannot_restart_four_audit_budget(self):
        image = Image.new('RGB', (100, 80))
        for backend_type in (Online, OnlineRepair):
            with self.subTest(backend=backend_type.__name__):
                vlm = EmptyQueryVLM()
                result = solve(backend_type(vlm), image, QUESTION, max_steps=3, repairs=3)
                self.assertEqual(vlm.audit_calls, 4)
                self.assertEqual(sum('audit' in event for event in result['trace']), 4)
                self.assertEqual(sum(event.get('event') == 'repeat_of_exhausted_question'
                                     for event in result['trace']), 3)
                self.assertFalse(any('parse_error' in event for event in result['trace']))
                self.assertEqual(len(result['ledger']), 1)
                self.assertEqual(result['status'], 'answered_with_uncertainty')

    def test_empty_conditional_deduction_still_continues_after_uncertain_visual_point(self):
        image = Image.new('RGB', (100, 80))
        for backend_type in (Online, OnlineRepair):
            with self.subTest(backend=backend_type.__name__):
                vlm = EmptyQueryVLM(conditional=True)
                result = solve(backend_type(vlm), image, QUESTION, max_steps=3, repairs=3)
                self.assertEqual(vlm.audit_calls, 5)
                self.assertEqual(result['answer'], 'B')
                self.assertEqual(result['status'], 'answered_with_uncertainty')
                self.assertEqual(result['ledger'][-1]['kind'], 'deduction')
                self.assertEqual(result['ledger'][-1]['evidence_question'], '')
                self.assertEqual(result['ledger'][-1]['verification'], 'uncertain')


if __name__ == '__main__':
    unittest.main()
