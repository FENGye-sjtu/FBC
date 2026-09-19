import json
import unittest
from types import SimpleNamespace

from efa.core import Policy, Scores
from efa.format import STEP_END, final_answer_contract
from efa.infer import run


class PassingBackend:
    image = SimpleNamespace(width=100, height=80)
    calls = []
    representation = 'fixture'

    def __init__(self, value):
        self.value = value

    def generate(self, *args, **kwargs):
        return [json.dumps(dict(atoms=[dict(subject='result', predicate='answer',
            value=self.value, family='attribute', kind='visual')], final=True)) + STEP_END]

    def route(self, query):
        return 'source'

    def score(self, claim, source):
        return Scores(0, 0, None, None, -1)


class FinalPremiseRejectingBackend(PassingBackend):
    def score(self, claim, source):
        return Scores(0, -10 if claim.kind == 'premise' else 0, None, None, -1)

    def refine(self, query, source, failures, round_id):
        return source

    def propose(self, query, source, k):
        return []

    def final_context(self, query, source):
        return source

    def rescore(self, query, source, values):
        return {value: -1 for value in values}


class FinalAnswerContractTests(unittest.TestCase):
    def test_yes_no_domain_is_deterministic(self):
        self.assertEqual(final_answer_contract('Is the first count larger?', 'yes'),
                         {'kind': 'yes_no', 'accepted': True})
        self.assertEqual(final_answer_contract('Is the first count larger?', '1'),
                         {'kind': 'yes_no', 'accepted': False})

    def test_locally_supported_count_cannot_be_verified_as_boolean_answer(self):
        result = run(PassingBackend('1'), 'Is the first count larger?',
                     {'default': Policy(-1, -1, use_deletion=False, use_specificity=False)})
        self.assertEqual(result['final']['answer'], '1')
        self.assertFalse(result['final']['verified'])
        self.assertEqual(result['final']['reason'], 'answer_contract_failure')

    def test_open_answer_can_still_be_verified(self):
        result = run(PassingBackend('blue'), 'What color is the bag?',
                     {'default': Policy(-1, -1, use_deletion=False, use_specificity=False)})
        self.assertTrue(result['final']['verified'])
        self.assertEqual(result['final']['answer_contract']['kind'], 'open')

    def test_local_visual_pass_does_not_replace_question_level_audit(self):
        result = run(FinalPremiseRejectingBackend('blue'), 'What color is the bag?',
                     {'default': Policy(-1, -1, use_deletion=False, use_specificity=False)})
        self.assertFalse(result['final']['verified'])
        self.assertEqual(result['final']['reason'], 'final_answer_audit_failed')
        self.assertEqual(result['final_audit']['resolution']['claim']['kind'], 'premise')


if __name__ == '__main__':
    unittest.main()
