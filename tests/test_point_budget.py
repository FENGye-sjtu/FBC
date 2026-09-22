import unittest

from trainingfree.core import Audit, Step, solve


class Scripted:
    def __init__(self, steps, audits):
        self.steps = iter(steps)
        self.audits = iter(audits)
        self.audit_calls = []
        self.feedback = []

    def propose(self, image, question, ledger, feedback):
        self.feedback.append(feedback)
        return next(self.steps)

    def audit(self, image, question, step, ledger):
        self.audit_calls.append(step)
        return next(self.audits)

    def finalize(self, image, question, ledger):
        return 'B'


class PointBudgetTests(unittest.TestCase):
    def test_twelve_audit_kind_switch_pathway_stops_after_four(self):
        query = 'What is the orientation of the front wheel?'
        # Previously each four-attempt block became a fresh step because only
        # visual candidates were checked against exhausted evidence questions.
        steps = ([Step('The wheel faces straight.', evidence_question=query)] * 4 +
                 [Step('The wheel therefore appears straight.', 'deduction',
                       evidence_question=query)] * 4 +
                 [Step('The orientation seems straight based on the image.', 'deduction',
                       evidence_question=query)] * 4)
        backend = Scripted(steps, [Audit('inconclusive')] * 12)
        result = solve(backend, None, 'q', max_steps=3, repairs=3)
        self.assertEqual(len(backend.audit_calls), 4)
        self.assertEqual(sum('audit' in e for e in result['trace']), 4)
        self.assertEqual(sum(e.get('event') == 'repeat_of_exhausted_question'
                             for e in result['trace']), 3)
        self.assertEqual(len(result['ledger']), 1)
        self.assertEqual(result['status'], 'answered_with_uncertainty')

    def test_empty_query_conditional_deduction_continues_with_uncertainty(self):
        steps = ([Step('The light may be red.', evidence_question='Light color?')] * 4 +
                 [Step('If the unresolved light is red, stopping is appropriate.', 'deduction',
                       evidence_question='', final_answer='B')])
        backend = Scripted(steps, [Audit('inconclusive')] * 4 + [Audit('supported')])
        result = solve(backend, None, 'q')
        self.assertEqual(len(backend.audit_calls), 5)
        self.assertEqual(result['answer'], 'B')
        self.assertEqual(result['status'], 'answered_with_uncertainty')
        self.assertEqual(result['ledger'][-1]['verification'], 'uncertain')

    def test_withheld_answer_moves_to_deduction_without_reauditing_fact(self):
        fact = Step('The water is right of the food.', evidence_question='Where is the water?',
                    final_answer='A')
        steps = [fact, fact, fact,
                 Step('The accepted relation matches option A.', 'deduction', final_answer='A')]
        backend = Scripted(steps, [Audit('supported', {'answer_withheld': True}), Audit('supported')])
        result = solve(backend, None, 'q')
        self.assertEqual(len(backend.audit_calls), 2)
        self.assertEqual(len(result['ledger']), 2)
        self.assertEqual(result['answer'], 'A')
        self.assertEqual(result['status'], 'verified_by_model')
        self.assertEqual(sum(e.get('event') == 'repeat_of_verified_point'
                             for e in result['trace']), 2)
        self.assertIn('empty evidence_question', backend.feedback[-1])

    def test_verified_fact_repetition_stops_after_two_redirections(self):
        fact = Step('The water is right of the food.', evidence_question='Where is the water?',
                    final_answer='A')
        backend = Scripted([fact] * 4, [Audit('supported', {'answer_withheld': True})])
        result = solve(backend, None, 'q')
        self.assertEqual(len(backend.audit_calls), 1)
        self.assertEqual(len(result['ledger']), 1)
        self.assertEqual(result['trace'][-1]['event'], 'redirection_exhausted')
        self.assertEqual(result['status'], 'answered_with_uncertainty')

    def test_distinct_evidence_question_remains_available(self):
        steps = ([Step('The light may be red.', evidence_question='What color is the light?')] * 4 +
                 [Step('The light is above the road.', evidence_question='Where is the light?',
                       final_answer='B')])
        backend = Scripted(steps, [Audit('inconclusive')] * 4 + [Audit('supported')])
        result = solve(backend, None, 'q')
        self.assertEqual(len(backend.audit_calls), 5)
        self.assertEqual(result['answer'], 'B')
        self.assertEqual(result['ledger'][-1]['verification'], 'verified')

    def test_success_on_third_repair_cannot_reopen_question_with_new_claim_or_kind(self):
        query = 'What color is the light?'
        initial = [Step(f'The light color is tentative observation {i}.',
                        evidence_question=query, final_answer='A') for i in range(4)]
        for kind in ('visual', 'deduction'):
            with self.subTest(kind=kind):
                reopened = [Step('The accepted light color should instead be reconsidered.', kind,
                                 evidence_question='WHAT color is the light!', final_answer='B')] * 3
                backend = Scripted(initial + reopened,
                                   [Audit('inconclusive')] * 3 +
                                   [Audit('supported', {'answer_withheld': True})])
                result = solve(backend, None, 'q', repairs=3)
                self.assertEqual(len(backend.audit_calls), 4)
                self.assertEqual(sum('audit' in event for event in result['trace']), 4)
                self.assertEqual(sum(event.get('event') == 'repeat_of_verified_question'
                                     for event in result['trace']), 3)
                self.assertEqual(len(result['ledger']), 1)
                self.assertEqual(result['ledger'][0]['claim'], initial[-1].claim)
                self.assertEqual(result['ledger'][0]['verification'], 'verified')
                self.assertEqual(result['status'], 'answered_with_uncertainty')

    def test_locked_verified_question_can_redirect_to_independent_evidence_and_deduction(self):
        query = 'What color is the light?'
        initial = [Step(f'The light color is tentative observation {i}.',
                        evidence_question=query, final_answer='A') for i in range(4)]
        steps = initial + [
            Step('A different light color is proposed.', evidence_question=query),
            Step('The road is clear.', evidence_question='Is the road clear?'),
            Step('The accepted color and clear road support option B.', 'deduction',
                 evidence_question='', final_answer='B'),
        ]
        backend = Scripted(steps, [Audit('inconclusive')] * 3 +
                           [Audit('supported', {'answer_withheld': True}),
                            Audit('supported'), Audit('supported')])
        result = solve(backend, None, 'q')
        self.assertEqual(len(backend.audit_calls), 6)
        self.assertEqual(result['answer'], 'B')
        self.assertEqual(len(result['ledger']), 3)
        self.assertEqual(result['ledger'][0]['claim'], initial[-1].claim)
        self.assertEqual(result['ledger'][-1]['evidence_question'], '')
        self.assertEqual(result['status'], 'verified_by_model')

    def test_empty_core_visual_queries_do_not_lock_all_unrelated_observations(self):
        # Production Online fills empty visual queries. Core-only backends with
        # unspecified queries keep their existing distinct-claim behavior.
        steps = [Step('A red light is visible.'), Step('The road is clear.'),
                 Step('The two observations support option B.', 'deduction', final_answer='B')]
        backend = Scripted(steps, [Audit('supported')] * 3)
        result = solve(backend, None, 'q')
        self.assertEqual(len(backend.audit_calls), 3)
        self.assertEqual(len(result['ledger']), 3)
        self.assertEqual(result['answer'], 'B')

    def test_failed_query_in_successful_repair_chain_is_closed_without_becoming_verified(self):
        failed = Step('The signal color is unclear.', evidence_question='Signal color?')
        accepted = Step('The lamp is red.', evidence_question='Lamp color?')
        reopened = Step('The signal is now claimed green.', evidence_question='SIGNAL color!',
                        final_answer='A')
        backend = Scripted([failed, accepted] + [reopened] * 3,
                           [Audit('inconclusive'), Audit('supported')])
        result = solve(backend, None, 'q')
        self.assertEqual(len(backend.audit_calls), 2)
        self.assertEqual(sum(event.get('event') == 'repeat_of_closed_question'
                             for event in result['trace']), 3)
        self.assertEqual(len(result['ledger']), 1)
        self.assertEqual(result['ledger'][0]['claim'], accepted.claim)
        self.assertEqual(result['ledger'][0]['evidence_question'], accepted.evidence_question)
        self.assertEqual(result['ledger'][0]['verification'], 'verified')
        self.assertNotIn(failed.claim, [step['claim'] for step in result['ledger']])
        self.assertIn('not accepted as verified', backend.feedback[-1])
        self.assertEqual(result['status'], 'answered_with_uncertainty')

    def test_closed_failed_query_redirect_allows_new_independent_question(self):
        steps = [Step('The signal color is unclear.', evidence_question='Signal color?'),
                 Step('The lamp is red.', evidence_question='Lamp color?'),
                 Step('The signal might be green.', 'deduction', evidence_question='Signal color?'),
                 Step('The road is clear.', evidence_question='Road clear?'),
                 Step('The accepted observations support option B.', 'deduction', final_answer='B')]
        backend = Scripted(steps, [Audit('inconclusive')] + [Audit('supported')] * 3)
        result = solve(backend, None, 'q')
        self.assertEqual(len(backend.audit_calls), 4)
        self.assertEqual(result['answer'], 'B')
        self.assertEqual([step['evidence_question'] for step in result['ledger']],
                         ['Lamp color?', 'Road clear?', ''])
        self.assertEqual(sum(event.get('event') == 'repeat_of_closed_question'
                             for event in result['trace']), 1)


if __name__ == '__main__':
    unittest.main()
