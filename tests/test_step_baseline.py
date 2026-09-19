import json
import unittest
from types import SimpleNamespace
from PIL import Image
from efa.format import STEP_END
from efa.step_baseline import run_step_only


class StepBaselineTests(unittest.TestCase):
    def test_unaudited_steps_keep_provisional_parents_without_source_calls(self):
        prompts=[]
        def generate(prompt,boxes,**kwargs):
            prompts.append(prompt)
            return [json.dumps(dict(atoms=[dict(subject='vehicle',predicate='category',value='train')],
                                    final=len(prompts)==3))+STEP_END]
        backend=SimpleNamespace(image=SimpleNamespace(width=100,height=80),calls=[],generate=generate)
        result=run_step_only(backend,'What vehicle?')
        self.assertEqual(result['final']['answer'],'train')
        self.assertFalse(result['final']['verified'])
        self.assertEqual(result['budget']['steps'],3)
        self.assertEqual(result['budget']['audits'],0)
        self.assertEqual(result['budget']['rounds'],0)
        self.assertIn('"verified": false',prompts[1])
        self.assertNotIn('"verified": true',''.join(prompts))

    def test_revisit_cannot_invoke_audit_in_disabled_path(self):
        backend=SimpleNamespace(image=SimpleNamespace(width=100,height=80),calls=[],
            generate=lambda *args,**kwargs:['{"review_parent":"step1.atom0@1"}'+STEP_END])
        result=run_step_only(backend,'Question')
        self.assertEqual(result['budget']['steps'],1)
        self.assertEqual(result['budget']['audits'],0)
        self.assertFalse(result['final']['verified'])
        self.assertEqual(result['final']['reason'],'invalid_or_unsupported_step')

    def test_complete_trajectory_preserves_assistant_history(self):
        histories=[]
        class Backend:
            image=Image.new('RGB',(20,10));calls=[]
            def generate_conversation(self,messages,max_tokens):
                histories.append([dict(message) for message in messages])
                number=len(histories)
                return json.dumps(dict(step=number,text=f'Reasoning {number}',atoms=[dict(
                    subject='object',predicate='property',value=str(number),family='attribute',kind='visual')],
                    final=number==2))+STEP_END
        result=run_step_only(Backend(),'Question',complete_trajectory=True)
        self.assertEqual([len(history) for history in histories],[1,3])
        self.assertEqual(histories[0][0]['content'].count('<image>'),1)
        self.assertNotIn('<image>',histories[1][-1]['content'])
        self.assertEqual(histories[1][1]['role'],'assistant')
        self.assertEqual((result['final']['answer'],result['budget']['steps']),('2',2))
