import hashlib
import json
import unittest

from PIL import Image

from trainingfree.core import Step, solve
from trainingfree.online import STEP_SCHEMA
from trainingfree.online_repair import (
    OnlineRepair, READ_SCHEMA, ATOMIC_READ_SCHEMA, ATOMIC_STEP_SCHEMA,
    claim_format_diagnostics,
)


QUESTION = 'Question: How many bars are visible?\nOptions:\nA. Two bars\nB. Three bars'
READBACK = {'observation': 'Three distinct horizontal members are visible.',
            'remaining_question': 'Which horizontal members have distinct endpoints?',
            'evidence_complete': False}
CANDIDATE = {'claim': 'Three distinct horizontal members are visible.', 'kind': 'visual',
             'evidence_question': 'Which horizontal members are visible?', 'final_answer': 'B'}


class ScriptedVLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, image, prompt, **kwargs):
        self.calls.append((image, prompt, kwargs))
        return json.dumps(next(self.responses))

    def yes_probability(self, image, prompt):
        return .1


class AtomicRepairTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (100, 80))

    def make_repair(self, **kwargs):
        vlm = ScriptedVLM([{'box': [10, 10, 80, 60]}, READBACK, CANDIDATE])
        backend = OnlineRepair(vlm, **kwargs)
        backend.last_candidate = Step('REJECTED_ASSERTION', evidence_question='REJECTED_QUERY')
        candidate = backend.propose(self.image, QUESTION, [], 'repair REJECTED_ASSERTION')
        return backend, vlm, candidate

    def test_default_reread_preserves_existing_prompt_and_schema_bytes(self):
        expected_prompts = [
            '69c2cafcdae1fc76a083fb4f84d5c91fdcb8133519cd09d216407812962d814c',
            'c7da2f461842dfbbaf0b06071de72cc443d03e6f9ea53123b61b338c3d2cf9ba',
            '20dfbbfae21991b1d71e2516e21e9c1a1a92fe95be6ff5f5dfc1e0adea64326e',
        ]
        expected_schemas = [
            '29203fbeaf35b3d57ef0fe41be51dcf1259f9ddde2e24dc40cafaf631f052b6e',
            'b98e8aabfaa483c916b3298ac3128ae16b0506e04179da4a5215dbc609ae9a7f',
            '1b0e413419b6454f8df080dd4c263999e5c14fbfe91a326872bb305ae435a30f',
        ]
        for kwargs in ({}, {'repair_style': 'reread'}):
            with self.subTest(kwargs=kwargs):
                backend, vlm, candidate = self.make_repair(**kwargs)
                self.assertEqual([hashlib.sha256(p.encode()).hexdigest() for _, p, _ in vlm.calls],
                                 expected_prompts)
                self.assertEqual([hashlib.sha256(json.dumps(k['schema'], sort_keys=True).encode()).hexdigest()
                                  for _, _, k in vlm.calls], expected_schemas)
                self.assertEqual([k['max_tokens'] for _, _, k in vlm.calls], [100, 320, 256])
                audit = backend.audit(self.image, QUESTION, candidate, [])
                self.assertNotIn('repair_style', audit.scores)
                self.assertNotIn('claim_format', audit.scores)

    def test_atomic_schema_is_independent_and_claim_remains_tentative(self):
        backend, vlm, candidate = self.make_repair(repair_style='atomic_v1')
        self.assertEqual(len(vlm.calls), 3)
        self.assertEqual(vlm.calls[1][2]['schema']['properties']['observation']['maxLength'], 180)
        self.assertEqual(vlm.calls[2][2]['schema']['properties']['claim']['maxLength'], 140)
        self.assertEqual(READ_SCHEMA['properties']['observation']['maxLength'], 600)
        self.assertEqual(STEP_SCHEMA['properties']['claim']['maxLength'], 300)
        self.assertIsNot(ATOMIC_READ_SCHEMA['properties'], READ_SCHEMA['properties'])
        self.assertIsNot(ATOMIC_STEP_SCHEMA['properties']['claim'], STEP_SCHEMA['properties']['claim'])
        self.assertIn('Answer ONLY the current neutral', vlm.calls[1][1])
        self.assertIn('Do not copy a multi-clause reading', vlm.calls[2][1])
        for _, prompt, _ in vlm.calls:
            self.assertNotIn('REJECTED_ASSERTION', prompt)
            self.assertNotIn('REJECTED_QUERY', prompt)
        self.assertEqual(candidate.verification, 'candidate')
        audit = backend.audit(self.image, QUESTION, candidate, [])
        self.assertEqual(audit.status, 'contradicted')
        self.assertEqual(audit.scores['repair_style'], 'atomic_v1')
        self.assertEqual(audit.scores['claim_format']['characters'], len(candidate.claim))
        self.assertEqual(audit.scores['claim_format']['schema_max_characters'], 140)

    def test_atomic_repairs_keep_four_audit_budget_and_explicit_uncertainty(self):
        class RepeatingVLM:
            def __init__(self):
                self.audit_calls = 0
                self.proposals = 0

            def generate(self, image, prompt, schema, **kwargs):
                if 'box' in schema['required']:
                    return json.dumps({'box': [10, 10, 80, 60]})
                if 'observation' in schema['required']:
                    return json.dumps(READBACK)
                self.proposals += 1
                return json.dumps(dict(CANDIDATE,
                                       claim=f'The bar count remains ambiguous in observation {self.proposals}.',
                                       evidence_question=''))

            def yes_probability(self, image, prompt):
                self.audit_calls += 1
                return .1

        vlm = RepeatingVLM()
        result = solve(OnlineRepair(vlm, repair_style='atomic_v1'), self.image, QUESTION, repairs=3)
        self.assertEqual(vlm.audit_calls, 4)
        self.assertEqual(result['status'], 'answered_with_uncertainty')
        self.assertEqual(result['ledger'][0]['verification'], 'uncertain')
        self.assertEqual(sum('audit' in event for event in result['trace']), 4)

    def test_truncation_signals_are_diagnostic_not_semantic_certificates(self):
        self.assertTrue(claim_format_diagnostics('The vehicle is separated from', 140)['truncation_suspected'])
        self.assertTrue(claim_format_diagnostics('x' * 139, 140)['truncation_suspected'])
        self.assertFalse(claim_format_diagnostics('The vehicle is on the left.', 140)['truncation_suspected'])

    def test_unknown_style_is_rejected(self):
        with self.assertRaises(ValueError):
            OnlineRepair(object(), repair_style='unknown')


if __name__ == '__main__':
    unittest.main()
