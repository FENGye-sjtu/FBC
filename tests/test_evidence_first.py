import json
import unittest
from unittest.mock import patch

from PIL import Image

from trainingfree.core import Audit, Step, solve
from trainingfree.evidence_first import (
    DEDUCTION_STEP_SCHEMA, EVIDENCE_STEP_SCHEMA, EvidenceFirstOnlineRepair,
)


QUESTION = ('Question: How many bars are visible?\nOptions:\n'
            'A. Two bars\nB. Three bars')


class FakeVLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, image, prompt, **kwargs):
        self.calls.append((image, prompt, kwargs))
        response = next(self.responses)
        return response if isinstance(response, str) else json.dumps(response)


def evidence():
    return {'claim': 'Three distinct bars are visible.', 'kind': 'visual',
            'evidence_question': 'How many distinct bars are visible?', 'final_answer': None}


def deduction():
    return {'claim': 'Three bars maps to option B.', 'kind': 'deduction',
            'evidence_question': '', 'final_answer': 'B'}


def incomplete_deduction():
    return {'claim': 'The verified count alone does not distinguish the remaining options.',
            'kind': 'deduction', 'evidence_question': '', 'final_answer': None}


class EvidenceFirstTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (100, 80))

    def test_initial_anchor_is_not_exposed_to_option_blind_evidence(self):
        vlm = FakeVLM(['A', evidence()])
        backend = EvidenceFirstOnlineRepair(vlm)
        step = backend.propose(self.image, QUESTION, [], '')
        self.assertEqual(backend.anchor_answer, 'A')
        self.assertIsNone(step.final_answer)
        _, evidence_prompt, kwargs = vlm.calls[1]
        self.assertNotIn('max_tokens', vlm.calls[0][2])
        self.assertNotIn('A. Two bars', evidence_prompt)
        self.assertNotIn('B. Three bars', evidence_prompt)
        self.assertNotIn('candidate answer', evidence_prompt.lower())
        self.assertIn('Grounding -> Observation', evidence_prompt)
        self.assertIn('claim must answer exactly that inspection question', evidence_prompt)
        self.assertIs(kwargs['schema'], EVIDENCE_STEP_SCHEMA)

    def test_verified_evidence_is_followed_by_separate_deduction(self):
        vlm = FakeVLM(['A', evidence(), deduction()])
        backend = EvidenceFirstOnlineRepair(vlm)
        first = backend.propose(self.image, QUESTION, [], '')
        ledger = [Step(first.claim, first.kind, first.evidence_question,
                       verification='verified')]
        second = backend.propose(self.image, QUESTION, ledger, '')
        self.assertEqual(second.kind, 'deduction')
        self.assertEqual(second.final_answer, 'B')
        self.assertIn('A. Two bars', vlm.calls[2][1])
        self.assertIs(vlm.calls[2][2]['schema'], DEDUCTION_STEP_SCHEMA)
        self.assertEqual(DEDUCTION_STEP_SCHEMA['properties']['final_answer']['type'],
                         ['string', 'null'])

    def test_insufficient_deduction_can_defer_answer_and_acquire_more_evidence(self):
        vlm = FakeVLM(['A', evidence(), incomplete_deduction(), evidence()])
        backend = EvidenceFirstOnlineRepair(vlm)
        first = backend.propose(self.image, QUESTION, [], '')
        ledger = [replace_verified(first)]
        second = backend.propose(self.image, QUESTION, ledger, '')
        self.assertIsNone(second.final_answer)
        ledger.append(replace_verified(second))
        third = backend.propose(self.image, QUESTION, ledger, '')
        self.assertEqual(third.kind, 'visual')
        self.assertIsNone(third.final_answer)
        self.assertNotIn('A. Two bars', vlm.calls[3][1])

    def test_failed_deduction_routes_to_new_option_blind_evidence(self):
        vlm = FakeVLM(['A', evidence(), deduction(), evidence()])
        backend = EvidenceFirstOnlineRepair(vlm)
        initial = backend.propose(self.image, QUESTION, [], '')
        ledger = [replace_verified(initial)]
        backend.propose(self.image, QUESTION, ledger, '')
        replacement = backend.propose(
            self.image, QUESTION, ledger,
            "Previous tentative claim 'Three bars maps to B' was not accepted")
        self.assertEqual(replacement.kind, 'visual')
        self.assertNotIn('Three bars maps to B', vlm.calls[3][1])
        self.assertNotIn('A. Two bars', vlm.calls[3][1])

    def test_answer_withheld_deduction_routes_forward_to_new_evidence(self):
        vlm = FakeVLM([evidence()])
        backend = EvidenceFirstOnlineRepair(vlm)
        backend.last_candidate = Step('Three bars maps to option B.', kind='deduction',
                                      final_answer='B')
        ledger = [Step('Three bars are visible.', verification='verified'),
                  Step('Three bars maps to option B.', kind='deduction',
                       verification='verified', final_answer=None)]
        step = backend.propose(self.image, QUESTION, ledger, '')
        self.assertEqual(step.kind, 'visual')
        self.assertIsNone(step.final_answer)
        self.assertIn('answer was withheld', vlm.calls[0][1])
        self.assertNotIn('A. Two bars', vlm.calls[0][1])

    def test_no_passed_deduction_preserves_direct_anchor(self):
        vlm = FakeVLM(['A', evidence()])
        backend = EvidenceFirstOnlineRepair(vlm)
        with patch.object(backend, 'audit', return_value=Audit('inconclusive', reason='unclear')):
            result = solve(backend, self.image, QUESTION, max_steps=1, repairs=0)
        self.assertEqual(result['answer'], 'A')
        self.assertEqual(result['status'], 'answered_with_uncertainty')
        self.assertEqual(backend.last_finalization['source'], 'tentative_direct_anchor')

    def test_negative_claim_does_not_use_nonidentifying_deletion_cf(self):
        vlm = FakeVLM([])
        backend = EvidenceFirstOnlineRepair(vlm, counterfactual=True)
        step = Step('There are no pedestrians visible.', kind='visual',
                    evidence_question='Are any pedestrians visible?')
        with patch('trainingfree.evidence_first.OnlineRepair.audit',
                   return_value=Audit('supported', {'support': .9}, 'support gate')) as gate:
            result = backend.audit(self.image, QUESTION, step, [])
        self.assertEqual(gate.call_count, 1)
        self.assertTrue(backend.counterfactual)
        self.assertTrue(result.scores['cf_skipped'])
        self.assertEqual(result.scores['cf_applicability'],
                         'not_identifying_for_negative_claim')

    def test_changed_anchor_needs_two_verified_visual_facts(self):
        backend = EvidenceFirstOnlineRepair(FakeVLM([]))
        backend.anchor_answer = 'B'
        changed = Step('The evidence maps to option A.', kind='deduction', final_answer='A')
        one_fact = [Step('The road appears uphill.', verification='verified')]
        with patch('trainingfree.evidence_first.OnlineRepair.audit', side_effect=[
                Audit('supported', {'entailment': .95}, 'passed'),
                Audit('supported', {'entailment': .95}, 'passed')]):
            withheld = backend.audit(self.image, QUESTION, changed, one_fact)
            allowed = backend.audit(self.image, QUESTION, changed, one_fact + [
                Step('A second independent slope cue is visible.', verification='verified')])
        self.assertTrue(withheld.scores['answer_withheld'])
        self.assertEqual(withheld.scores['anchor_disagreement']['required_visual_n'], 2)
        self.assertNotIn('answer_withheld', allowed.scores)


def replace_verified(step):
    return Step(step.claim, step.kind, step.evidence_question,
                verification='verified')


if __name__ == '__main__':
    unittest.main()
