import json
import unittest

from PIL import Image

from trainingfree.core import Step, solve
from trainingfree.online_repair import OnlineRepair, READ_SCHEMA, question_stem


QUESTION = 'Question: How many bars are visible?\nOptions:\nA. Two bars\nB. Three bars'


class FakeVLM:
    def __init__(self, responses, support_scores=None):
        self.responses = iter(responses)
        self.calls = []
        self.support_scores = iter(support_scores) if support_scores is not None else None

    def generate(self, image, prompt, **kwargs):
        self.calls.append((image, prompt, kwargs))
        return json.dumps(next(self.responses))

    def yes_probability(self, image, prompt):
        return next(self.support_scores) if self.support_scores is not None else .1


def readback(remaining='Which horizontal members have distinct endpoints?'):
    return {'observation': 'Three distinct horizontal members are visible.',
            'remaining_question': remaining, 'evidence_complete': False}


def candidate():
    return {'claim': 'Three distinct horizontal members are visible.', 'kind': 'visual',
            'evidence_question': 'Which horizontal members are visible?', 'final_answer': 'B'}


class OnlineRepairTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (100, 80))

    def test_acquisition_and_repair_do_not_copy_rejected_claim(self):
        vlm = FakeVLM([{'box': [10, 10, 80, 60]}, readback(), candidate()])
        backend = OnlineRepair(vlm)
        backend.last_candidate = Step('REJECTED_ASSERTION_TOKEN', evidence_question='REJECTED_QUERY_TOKEN')
        ledger = []
        step = backend.propose(self.image, QUESTION, ledger,
                               'Previous tentative claim REJECTED_ASSERTION_TOKEN was not accepted')
        self.assertEqual(step.final_answer, 'B')
        self.assertEqual(ledger, [])
        for _, prompt, _ in vlm.calls:
            self.assertNotIn('REJECTED_ASSERTION_TOKEN', prompt)
            self.assertNotIn('REJECTED_QUERY_TOKEN', prompt)
        self.assertNotIn('Two bars', vlm.calls[0][1])
        self.assertNotIn('Two bars', vlm.calls[1][1])
        self.assertIn('TENTATIVE', vlm.calls[2][1])
        self.assertEqual(vlm.calls[1][2]['schema'], READ_SCHEMA)
        # A reread cannot turn a failed normal audit into an accepted step.
        audit = backend.audit(self.image, QUESTION, step, ledger)
        self.assertEqual(audit.status, 'contradicted')
        self.assertFalse(audit.scores['repair_readback']['evidence_complete'])

    def test_second_repair_targets_missing_cue_third_restores_full_context(self):
        vlm = FakeVLM([{'box': [10, 10, 80, 60]}, readback(), candidate(),
                       {'box': [20, 20, 90, 70]}, readback(''), candidate(),
                       readback(''), candidate()])
        backend = OnlineRepair(vlm)
        backend.last_candidate = Step('An unsupported observation')
        backend.propose(self.image, QUESTION, [], 'repair')
        backend.propose(self.image, QUESTION, [], 'repair')
        self.assertIn('Which horizontal members have distinct endpoints?', vlm.calls[3][1])
        backend.propose(self.image, QUESTION, [], 'repair')
        self.assertIs(vlm.calls[6][0], self.image)
        self.assertIs(vlm.calls[7][0], self.image)
        self.assertEqual(backend.pending_readback['repair_round'], 3)
        self.assertEqual(backend.pending_readback['view'], 'full')
        self.assertEqual(backend.last_candidate.box, ())

    def test_successful_initial_proposal_does_not_pay_for_reread(self):
        vlm = FakeVLM([candidate()])
        backend = OnlineRepair(vlm)
        backend.propose(self.image, QUESTION, [], '')
        self.assertEqual(len(vlm.calls), 1)
        self.assertIsNone(backend.pending_readback)

    def test_invalid_reread_does_not_become_a_candidate(self):
        vlm = FakeVLM([{'box': [10, 10, 80, 60]}, {'observation': ''}])
        backend = OnlineRepair(vlm)
        original = Step('An unsupported observation')
        backend.last_candidate = original
        with self.assertRaises(ValueError):
            backend.propose(self.image, QUESTION, [], 'repair')
        self.assertIs(backend.last_candidate, original)

    def test_question_stem_removes_choices_from_acquisition(self):
        self.assertEqual(question_stem(QUESTION), 'How many bars are visible?')

    def test_corrected_tentative_answer_beats_weaker_initial_without_becoming_verified(self):
        initial = dict(candidate(), claim='Two horizontal members are visible.', final_answer='A')
        vlm = FakeVLM([initial, {'box': [10, 10, 80, 60]}, readback(), candidate()], [.1, .6])
        backend = OnlineRepair(vlm)
        result = solve(backend, self.image, QUESTION, max_steps=1, repairs=1)
        self.assertEqual(result['answer'], 'B')
        self.assertEqual(result['status'], 'answered_with_uncertainty')
        self.assertTrue(all(step['verification'] == 'uncertain' for step in result['ledger']))
        self.assertEqual(backend.last_finalization['source'], 'highest_support_tentative_candidate')
        self.assertEqual(backend.last_finalization['support'], .6)

    def test_tentative_score_tie_uses_latest_and_resets_on_new_case(self):
        vlm = FakeVLM([candidate()], [.6, .6])
        backend = OnlineRepair(vlm)
        backend.audit(self.image, QUESTION, Step('Two members are visible.', final_answer='A'), [])
        backend.audit(self.image, QUESTION, Step('Three members are visible.', final_answer='B'), [])
        self.assertEqual(backend.finalize(self.image, QUESTION, []), 'B')
        backend.propose(self.image, QUESTION, [], '')
        self.assertEqual(backend.audited_candidates, [])
        self.assertIsNone(backend.last_finalization)


if __name__ == '__main__':
    unittest.main()
