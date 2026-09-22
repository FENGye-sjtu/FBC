import json
import unittest
from unittest.mock import patch

from PIL import Image

from trainingfree.core import Audit, Step, solve
from trainingfree.online import Online, PROPOSE, STEP_SCHEMA
from trainingfree.online_repair import OnlineRepair
from trainingfree.prompts import answer_format, compact_initial_prompt, task_question


QUESTION = ('Question: How many bars are visible?\nOptions:\nA. Two bars\nB. Three bars\n'
            'Please select the correct answer from the options above.')


def candidate(answer='B'):
    return {'claim':'Three distinct horizontal bars are visible.', 'kind':'visual',
            'evidence_question':'How many distinct horizontal bars are visible?',
            'final_answer':answer}


class FakeVLM:
    def __init__(self,responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self,image,prompt,**kwargs):
        self.calls.append((image,prompt,kwargs))
        return json.dumps(next(self.responses))


class InitialPromptTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB',(100,80))

    def test_default_prompt_is_byte_for_byte_full_prompt(self):
        vlm = FakeVLM([candidate()])
        backend = Online(vlm)
        backend.propose(self.image,QUESTION,[],'')
        expected = PROPOSE % (task_question(QUESTION),'[]','')
        expected += '\nFor final_answer only: ' + answer_format(QUESTION)
        self.assertEqual(vlm.calls[0][1],expected)
        self.assertEqual(backend.initial_style,'full')

    def test_compact_first_step_retains_options_and_generation_budget(self):
        vlm = FakeVLM([candidate(None)])
        backend = Online(vlm,initial_style='compact_v1')
        step = backend.propose(self.image,QUESTION,[],'')
        image,prompt,kwargs = vlm.calls[0]
        self.assertIs(image,self.image)
        self.assertEqual(prompt,compact_initial_prompt(QUESTION))
        self.assertIn('A. Two bars\nB. Three bars',prompt)
        self.assertIn('use one of A, B; otherwise use JSON null',prompt)
        self.assertIn('- kind: "visual"',prompt)
        self.assertIn('must pass the audit',prompt)
        self.assertNotIn('Evidence ledger',prompt)
        self.assertEqual(kwargs['max_tokens'],256)
        self.assertIs(kwargs['schema'],STEP_SCHEMA)
        self.assertIsNone(step.final_answer)

    def test_compact_does_not_change_repair_or_nonempty_ledger_prompt(self):
        vlm = FakeVLM([candidate(),{'box':[10,10,80,60]},candidate(),candidate()])
        backend = Online(vlm,initial_style='compact_v1')
        backend.propose(self.image,QUESTION,[],'')
        backend.propose(self.image,QUESTION,[],'Correct this unsupported candidate.')
        repair_prompt = vlm.calls[2][1]
        self.assertTrue(repair_prompt.startswith('Solve the visual question incrementally.'))
        self.assertIn('Repair instruction (not an accepted premise): Correct this unsupported candidate.',
                      repair_prompt)
        self.assertEqual(vlm.calls[2][2]['max_tokens'],256)
        self.assertIs(vlm.calls[2][2]['schema'],STEP_SCHEMA)
        ledger = [Step('A horizontal bar is visible.',verification='verified')]
        backend.propose(self.image,QUESTION,ledger,'')
        expected = PROPOSE % (task_question(QUESTION),backend.ledger_text(ledger),'')
        expected += '\nFor final_answer only: ' + answer_format(QUESTION)
        self.assertEqual(vlm.calls[3][1],expected)

    def test_parse_retry_with_empty_ledger_still_uses_full_instructions(self):
        vlm = FakeVLM([candidate()])
        backend = Online(vlm,initial_style='compact_v1')
        backend.propose(self.image,QUESTION,[],'Invalid step format: missing claim.')
        self.assertTrue(vlm.calls[0][1].startswith('Solve the visual question incrementally.'))

    def test_compact_candidate_cannot_bypass_failed_audit(self):
        vlm = FakeVLM([candidate()])
        backend = Online(vlm,initial_style='compact_v1')
        with patch.object(backend,'audit',return_value=Audit('inconclusive',reason='unclear')) as gate, \
             patch.object(backend,'finalize',return_value='B'):
            result = solve(backend,self.image,QUESTION,max_steps=1,repairs=0)
        self.assertEqual(gate.call_count,1)
        self.assertEqual(result['status'],'answered_with_uncertainty')
        self.assertEqual(result['ledger'][0]['verification'],'uncertain')

    def test_validation_and_online_repair_inheritance(self):
        with self.assertRaisesRegex(ValueError,'Unknown initial proposal style'):
            Online(FakeVLM([]),initial_style='unknown')
        vlm = FakeVLM([candidate()])
        backend = OnlineRepair(vlm,initial_style='compact_v1')
        backend.propose(self.image,QUESTION,[],'')
        self.assertEqual(vlm.calls[0][1],compact_initial_prompt(QUESTION))
        self.assertIsNone(backend.pending_readback)


if __name__ == '__main__':
    unittest.main()
