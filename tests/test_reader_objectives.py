import math
from types import SimpleNamespace
import unittest
try:
    import torch
except ImportError:
    torch=None
from efa.reader_objectives import backward_objectives


@unittest.skipIf(torch is None,'Optional torch extra is unavailable')
class ReaderObjectiveTests(unittest.TestCase):
    def test_sequence_losses_and_gradients_keep_whole_string_probability(self):
        logits=torch.tensor([.3,-.4,.7],requires_grad=True);calls=[]
        def sequence(prompt,boxes,value):
            calls.append((prompt,value))
            index={'cat':0,'object beside bench':1,'dog':2}[value]
            # Unequal suffix lengths expose accidental length normalization.
            return torch.nn.functional.logsigmoid(logits[index]),[7,5,3][index]
        backend=SimpleNamespace(image=SimpleNamespace(width=10,height=10),sequence_logp=sequence)
        row=dict(query=dict(subject='object beside bench',predicate='category'),answer='cat',
            identity='object beside bench',boxes=[[0,0,8,8]],context_boxes=[[0,0,8,8],[0,0,10,10]],
            negatives=[dict(value='dog')])
        metrics=backward_objectives(backend,row,identity_weight=.2,negative_weight=1)
        self.assertAlmostEqual(metrics['read'],math.log1p(math.exp(-.3)),places=6)
        self.assertAlmostEqual(metrics['identity'],math.log1p(math.exp(.4)),places=6)
        self.assertAlmostEqual(metrics['negative'],math.log1p(math.exp(.7)),places=6)
        self.assertLess(logits.grad[0].item(),0)
        self.assertLess(logits.grad[1].item(),0)
        self.assertGreater(logits.grad[2].item(),0)
        self.assertEqual(calls[0][0],calls[2][0])
        self.assertNotRegex(calls[0][0],r'\bcat\b')
        self.assertNotRegex(calls[2][0],r'\bdog\b')
