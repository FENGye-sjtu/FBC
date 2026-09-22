import json
from pathlib import Path
import tempfile
import unittest

from trainingfree.answers import PARSER_VERSION, _extract_v2, answer_source, choices_from_question, extract_answer, raw_answer


QUESTION = ('Question: What color is the car?\nOptions:\n'
            'A. red\nB. blue\nC. green\n'
            'Please select the correct answer from the options above.')


class AnswerTests(unittest.TestCase):
    def test_v2_inline_affirmative_choices_preserve_explicit_label(self):
        question = 'A. 25\nB. 35\nC. 45'
        for text in ('The speed limit on this road is B. 35',
                     'Thus, the speed limit on this road is B.35',
                     'Final answer: the speed limit is (B). 35',
                     'The selected speed is option B: 35',
                     'The chosen speed would be B.35'):
            with self.subTest(text=text):
                self.assertEqual(extract_answer(text, question), 'B')
        # Same rule for labels conflicting with public wording; never use the
        # correct option or option-value matching to override the explicit A.
        for question in ('A. 25\nB. 35\nC. 45', 'A. 35\nB. 25\nC. 45'):
            self.assertEqual(extract_answer('The speed limit is A.35', question), 'A')
        self.assertEqual(extract_answer('Therefore, the direction is C. left',
                                       'A. left\nB. right\nC. straight'), 'C')

    def test_v2_inline_rejections_and_ambiguity(self):
        question = 'A. 25\nB. 35\nC. 45'
        for text in ('The speed is not B.35', 'If the speed is B.35, the sign is clear.',
                     'Assuming the speed is B.35, proceed.', 'It is false that the speed is B.35',
                     'The speed is B.35 or C.45', 'The speed is B.35 or C',
                     'The speed is B.35 and the other answer is C.45',
                     'The speed is B.35/C', 'The rejected speed is B.35',
                     'The speed is B.35 but that is incorrect.',
                     'Example: the speed is B.35'):
            with self.subTest(text=text):
                self.assertNotIn(extract_answer(text, question), ('A', 'B', 'C'))

    def test_runtime_and_suite_share_raw_source_precedence(self):
        from trainingfree import run, evaluation_suite
        self.assertIs(run.raw_answer, raw_answer)
        self.assertIs(evaluation_suite.raw_answer, raw_answer)
        self.assertIs(evaluation_suite.answer_source, answer_source)
        for row, source, value in (
                ({'raw': 'FINAL ANSWER: B', 'answer': 'A'}, 'raw', 'FINAL ANSWER: B'),
                ({'raw': '', 'answer': 'A'}, 'raw', ''),
                ({'raw': None, 'answer': 'B'}, 'answer', 'B'),
                ({'answer': 'B'}, 'answer', 'B'), ({'raw': None, 'answer': None}, 'missing', '')):
            self.assertEqual(answer_source(row), source)
            self.assertEqual(raw_answer(row), value)

    def test_v3_weak_conclusion_with_complete_terminal_option(self):
        self.assertEqual(PARSER_VERSION, 'committed_answer_v3')
        question = ('If I stay in this lane, where do I end up?\n'
                    'A. Highway 1\nB. Highway 95\nC. Highway 280')
        for text in ('Thus, if you stay in this lane, you will end up on:\n\nB. Highway 95',
                     'Therefore, the route is:\n\n**B. Highway 95**',
                     'The destination is:\n\n(B) Highway 95',
                     'Thus, staying in this lane leads to:\n\nB: Highway 95.'):
            with self.subTest(text=text):
                self.assertEqual(extract_answer(text, question), 'B')
        # A negative phrase belonging to an actual choice is still valid.
        self.assertEqual(extract_answer('Thus, the available evidence supports:\n\nB. Not enough information',
                                       'A. Yes\nB. Not enough information'), 'B')

    def test_v3_terminal_option_cannot_override_stronger_answers(self):
        question = 'A. Highway 1\nB. Highway 95\nC. Highway 280'
        for text in ('FINAL ANSWER: A\nAdditional discussion.\nB. Highway 95',
                     '{"final_answer":"A"}\nB. Highway 95',
                     '<final_answer>A</final_answer>\nB. Highway 95'):
            self.assertEqual(extract_answer(text, question), 'A')
        self.assertNotEqual(extract_answer('Thus, the route is:\n\nA. Highway 95', question), 'B')
        self.assertEqual(extract_answer('FINAL ANSWER: A. Highway 95', question), 'A')

    def test_v3_rejected_conditional_and_multiple_terminal_candidates(self):
        question = 'A. Highway 1\nB. Highway 95\nC. Highway 280'
        for text in ('Thus, the alternatives are:\n\nA. Highway 1\nB. Highway 95',
                     'Thus, the rejected option is:\n\nB. Highway 95',
                     'Thus, if signs meant another route, the answer might be:\n\nB. Highway 95',
                     'Thus, the route is:\n\nB. Highway 95 is not correct',
                     'Thus, the route is:\n\nB. Highway 95 if the sign means something else',
                     'Thus, the route is:\n\nB. Highway 95 or C'):
            with self.subTest(text=text):
                self.assertNotIn(extract_answer(text, question), ('A', 'B', 'C'))

    def test_v3_is_additive_even_for_known_v2_false_positives(self):
        question = 'A. Highway 1\nB. Highway 95\nC. Highway 280'
        # These are v2 limitations, not newly accepted v3 formats. The additive
        # extension must not silently rewrite an already published v2 label.
        inherited = ('Thus, this is not the selected answer:\n\nB. Highway 95',
                     'Possible candidates:\nA. Highway 1\nB. Highway 95')
        ordinary = ('The most reasonable estimate would be:\n\nB. Highway 95',
                    'FINAL ANSWER: A. Highway 95', 'B', 'Answer: C')
        for text in inherited + ordinary:
            with self.subTest(text=text):
                previous = _extract_v2(text, question)
                self.assertIn(previous, choices_from_question(question))
                self.assertEqual(extract_answer(text, question), previous)
        for label in 'ABC':
            for word in ('Highway 1', 'Highway 95', 'Highway 280'):
                text = f'Final answer: {label}. {word}'
                self.assertEqual(extract_answer(text, question), _extract_v2(text, question))

    def test_option_formats_and_prompt_suffix(self):
        expected = {'A': 'red', 'B': 'blue', 'C': 'green'}
        self.assertEqual(choices_from_question(QUESTION), expected)
        self.assertEqual(choices_from_question('A. red B. blue C. green'), expected)
        self.assertEqual(choices_from_question('(A) red\n(B) blue\n(C) green'), expected)

    def test_generous_committed_formats(self):
        for raw in ('B', '(b)', '[B]', '**B**', 'B\n\nThe car is blue.',
                    'B. Blue. The reflection is visible.', 'Option B',
                    'I choose B',
                    'B is the correct answer.', 'Option B is correct.',
                    'Final answer: "B"', 'Final answer: (B) — blue',
                    '**Final answer:** B', 'Final answer is B.',
                    'The correct answer is **B** because it is blue.',
                    '<final_answer>B</final_answer>', '<answer>blue</answer>',
                    '```json\n{"reasoning": "A was considered", "answer": "B"}\n```',
                    r'\boxed{B}', r'\boxed{\text{B}}',
                    'Therefore, the color is blue.', 'Final answer: blue',
                    'Thus, the final answer is:\nB. blue',
                    'Given these observations, the correct answer is:\n\nB. blue\n\nA is rejected.'):
            with self.subTest(raw=raw):
                self.assertEqual(extract_answer(raw, QUESTION), 'B')

    def test_explicit_final_beats_explanatory_terminal_letter(self):
        self.assertEqual(extract_answer('Final answer: B\nAn alternative discussed was\nA', QUESTION), 'B')
        self.assertEqual(extract_answer('Final answer: A\nCorrection: Final answer: B', QUESTION), 'B')
        self.assertEqual(extract_answer('Final answer: A. blue', QUESTION), 'A')
        self.assertEqual(extract_answer('Final answer: (A) — blue', QUESTION), 'A')
        self.assertEqual(extract_answer('Final answer: A - blue', QUESTION), 'A')

    def test_no_ground_truth_or_reference_input(self):
        # A declared label is respected even if the option wording disagrees;
        # a scorer must not search for a reference-compatible interpretation.
        swapped = QUESTION.replace('A. red\nB. blue', 'A. blue\nB. red')
        for question in (QUESTION, swapped):
            self.assertEqual(extract_answer('Final answer: A. blue', question), 'A')

    def test_mentions_negations_and_alternatives_are_not_answers(self):
        for raw in ('The options are red, blue, and green.',
                    'I considered blue.', 'Blue is not visible.',
                    'There is no blue car.', 'The car is not blue.',
                    'If the car were blue, B would be correct.',
                    'If the answer is B, we should see a blue car.',
                    'If B is the correct answer, the car is blue.',
                    'Final answer: A or B', 'Final answer: A/B', 'Final answer: B, or A',
                    'Final answer: either blue or red',
                    'Final answer: not blue', 'Final answer: B is not correct',
                    'A: red B: blue', 'A blue car is one possible interpretation.',
                    'I reject answer: B', 'Format: Final answer: B'):
            with self.subTest(raw=raw):
                self.assertNotIn(extract_answer(raw, QUESTION), ('A', 'B', 'C'))

    def test_multiple_uncommitted_labels_remain_ambiguous(self):
        self.assertEqual(extract_answer('A\nB', QUESTION), 'A\nB')
        self.assertEqual(extract_answer('A\nC\nB', QUESTION), 'A\nC\nB')
        self.assertEqual(extract_answer('The car seems red.\nTherefore, it seems blue.\n\nC',
                                       QUESTION), 'C')

    def test_final_abstention_does_not_recover_an_earlier_guess(self):
        for raw in ('B\nReasoning.\nFinal answer: N/A',
                    'Answer: B\nFinal answer: cannot determine',
                    'Final answer: none of the options; blue is not visible.'):
            with self.subTest(raw=raw):
                self.assertNotIn(extract_answer(raw, QUESTION), ('A', 'B', 'C'))

    def test_option_text_overlap_and_negation(self):
        question = 'A. left\nB. bottom left\nC. right'
        self.assertEqual(extract_answer('Final answer: bottom left', question), 'B')
        self.assertEqual(extract_answer('Thus, it is at the bottom left.', question), 'B')
        self.assertEqual(extract_answer('Final answer: not red but blue', QUESTION), 'B')
        self.assertEqual(extract_answer('Final answer: not visible',
                                       'A. visible\nB. not visible'), 'B')

    def test_unambiguous_lexical_equivalents(self):
        self.assertEqual(extract_answer('Final answer: two', 'A. 1\nB. 2\nC. 3'), 'B')
        self.assertEqual(extract_answer('There are zero cars.', 'A. 0 cars\nB. 1 car'), 'A')
        self.assertEqual(extract_answer('Final answer: yep', 'A. No\nB. Yes'), 'B')
        self.assertEqual(extract_answer('Nope', 'A. No\nB. Yes'), 'A')
        self.assertNotIn(extract_answer('Final answer: true', 'A. No\nB. Yes'), ('A', 'B'))
        self.assertNotIn(extract_answer('two', 'A. 2\nB. two\nC. 3'), ('A', 'B'))
        self.assertEqual(extract_answer('E', QUESTION), 'E')

    def test_open_answer_compatibility_and_empty(self):
        self.assertEqual(extract_answer(None), '')
        self.assertEqual(extract_answer(' \n'), '')
        self.assertEqual(extract_answer('Therefore, ', QUESTION), '')
        self.assertEqual(extract_answer('Final answer: ', QUESTION), '')
        self.assertEqual(extract_answer('Thus, Yes.'), 'Yes')
        self.assertEqual(extract_answer('Reasoning\nFinal answer:\n\nGreen'), 'Green')
        self.assertEqual(extract_answer('A: a label in open text', 'Describe the label'),
                         'A: a label in open text')

    def test_audit_records_gains_and_losses(self):
        from trainingfree.parser_audit import audit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, raw = root / 'cases.jsonl', root / 'raw.jsonl'
            dataset.write_text('\n'.join(json.dumps(row) for row in (
                {'id': 1, 'question': QUESTION, 'reference': 'B'},
                {'id': 2, 'question': QUESTION, 'reference': 'A'})), encoding='utf-8')
            raw.write_text('\n'.join(json.dumps(row) for row in (
                {'id': 1, 'raw': 'Final answer: B'},
                {'id': 2, 'raw': 'Final answer: C'})), encoding='utf-8')
            result = audit([raw], dataset, legacy_extract=lambda text, question: 'A')['files'][0]
            self.assertEqual(result['recovered_ids'], [1])
            self.assertEqual(result['lost_ids'], [2])
            self.assertEqual(len(result['disagreements']), 2)
            self.assertEqual(result['old_correct'], result['new_correct'])
            self.assertEqual(len(result['sha256']), 64)

    def test_audit_uses_raw_precedence_and_records_actual_legacy_source(self):
        import hashlib
        import inspect
        from trainingfree.parser_audit import audit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, predictions = root / 'cases.jsonl', root / 'raw.jsonl'
            dataset.write_text(json.dumps({'id': 1, 'question': QUESTION, 'reference': 'B'}))
            predictions.write_text(json.dumps({'id': 1, 'raw': '', 'answer': 'B'}))
            report = audit([predictions], dataset, legacy_extract=extract_answer)
            self.assertEqual(report['files'][0]['new_correct'], 0)
            self.assertEqual(report['legacy_parser_version'], PARSER_VERSION)
            self.assertEqual(report['legacy_source_sha256'], hashlib.sha256(
                Path(inspect.getsourcefile(extract_answer)).read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
