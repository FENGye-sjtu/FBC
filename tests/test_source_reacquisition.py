import json
import unittest
from unittest.mock import Mock

from PIL import Image

from efa.core import Budget, Claim, Policy, Scores, resolve
from efa.grammar import ROUTE_SCHEMA
from efa.latent_backend import LatentBackend
from efa.qwen import QwenBackend, Sources
from efa.regions import Box


class SourceReacquisitionTests(unittest.TestCase):
    old_target=Box(0,0,10,10)
    new_target=Box(40,30,80,90)

    def backend(self,kind=QwenBackend,enabled=None,failed_route=False):
        backend=kind.__new__(kind)
        backend.image=Image.new('RGB',(100,100))
        backend.image_digest='fixture'
        backend.encoder_config={'model':'fixture'}
        backend.premises=[{'claim':{'value':'secret_previous_answer'}}]
        if enabled is not None:backend.reacquire_on_binding_failure=enabled
        first=json.dumps(dict(boxes=[[0,0,10,10]],context_boxes=[[0,0,20,20]]))
        second='invalid route' if failed_route else json.dumps(
            dict(boxes=[[40,30,80,90]],context_boxes=[[20,10,95,95]]))
        backend.generate=Mock(side_effect=[[first]]+[[second]]*3)
        backend.propose=Mock(return_value=['elephant','horse','cow'])
        # Binding depends on selecting the right pixels. Even high numerical
        # scores cannot certify a route whose geometry was invalid.
        backend.score=Mock(side_effect=lambda claim,source: Scores(
            0 if source.target!=(self.old_target,) else -10,0,None,None,-1,source.valid))
        backend.rescore=Mock(return_value={'elephant':-1,'horse':-2,'cow':-3})
        return backend

    def test_binding_failure_replaces_target_without_exposing_value_or_trace(self):
        for kind in (QwenBackend,LatentBackend):
            with self.subTest(backend=kind.__name__):
                backend=self.backend(kind,enabled=True)
                budget=Budget()
                result=resolve(Claim('animal behind the tree','species','elephant'),
                               backend,Policy(-1,-1),budget)
                self.assertTrue(result.local_pass)
                self.assertEqual((result.rounds,budget.rounds,budget.audits),(1,1,4))
                self.assertEqual(result.claim.value,'elephant')
                self.assertEqual(result.events[0]['failures'],['binding'])
                self.assertEqual(result.events[1]['source']['target'],[vars(self.new_target)])
                self.assertEqual(backend.generate.call_count,2)
                retry=backend.generate.call_args
                prompt,boxes=retry.args
                self.assertIn('replace the target boxes',prompt)
                self.assertIn('context_boxes',prompt)
                self.assertIn('[[0, 0, 10, 10]]',prompt)
                for excluded in ('elephant','horse','cow','secret_previous_answer'):
                    self.assertNotIn(excluded,prompt)
                self.assertEqual(boxes,[Box(0,0,100,100)])
                self.assertEqual(retry.kwargs,{'json_schema':ROUTE_SCHEMA})
                self.assertEqual(backend.propose.call_args.args[2],3)

    def test_default_preserves_targets_and_existing_three_round_budget(self):
        backend=self.backend()
        self.assertFalse(backend.reacquire_on_binding_failure)
        budget=Budget()
        result=resolve(Claim('animal behind the tree','species','elephant'),
                       backend,Policy(-1,-1),budget)
        self.assertFalse(result.local_pass)
        self.assertEqual((result.rounds,budget.rounds,budget.audits),(3,3,10))
        self.assertEqual(backend.generate.call_count,1)
        self.assertEqual(backend.propose.call_count,3)
        for call in backend.propose.call_args_list:
            self.assertEqual(call.args[1].target,(self.old_target,))
            self.assertEqual(call.args[2],3)
        first_context=backend.propose.call_args_list[0].args[1].context
        self.assertEqual(first_context,(Box(0,0,20,20).expand(100,100,.25),))

    def test_failed_reacquisition_stays_invalid_and_cannot_pass(self):
        backend=self.backend(enabled=True,failed_route=True)
        budget=Budget()
        result=resolve(Claim('animal behind the tree','species','elephant'),
                       backend,Policy(-1,-1),budget)
        self.assertFalse(result.local_pass)
        self.assertEqual((result.rounds,budget.rounds,budget.audits),(3,3,10))
        self.assertEqual(backend.generate.call_count,4)  # initial + one per round
        self.assertEqual(backend.propose.call_count,3)
        for event in result.events[1:-1]:
            self.assertFalse(event['source']['valid'])
            self.assertIn('invalid',event['failures'])
        self.assertFalse(result.events[-1]['source']['valid'])

    def test_only_binding_or_invalid_reacquires_and_premises_never_route(self):
        backend=self.backend(enabled=True)
        backend.audit_kind='visual'
        query={'subject':'animal behind the tree','predicate':'species'}
        source=Sources((self.old_target,),(self.old_target,),True)
        expanded=backend.refine(query,source,['support'],1)
        self.assertEqual(expanded.target,source.target)
        backend.generate.assert_not_called()
        backend.audit_kind='premise'
        self.assertIs(backend.refine(query,source,['binding','invalid'],1),source)
        backend.generate.assert_not_called()


if __name__=='__main__':unittest.main()
