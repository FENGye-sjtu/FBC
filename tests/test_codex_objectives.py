import math
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

from efa.codex_objectives import attach_reviewed_replay, backward_codex_objective, LABEL_BASIS


class CorrectiveReplayTests(unittest.TestCase):
    def test_corrections_cannot_enter_a_different_image_or_unreviewed_parent(self):
        read = dict(episode_id='e', image_sha256='a', annotation_status='reviewed_episode', label_basis=LABEL_BASIS)
        pair = dict(parent_episode_id='e', image_sha256='b')
        with self.assertRaisesRegex(ValueError, 'accepted training image'):
            attach_reviewed_replay([pair], [read])
        pair['image_sha256'] = 'a'
        self.assertEqual(len(attach_reviewed_replay([pair], [read])), 2)
        read['annotation_status'] = 'teacher_reviewed_episode'
        with self.assertRaisesRegex(ValueError, 'individually adjudicated'):
            attach_reviewed_replay([pair], [read])


@unittest.skipIf(torch is None, 'Optional torch extra is unavailable')
class CorrectiveGradientTests(unittest.TestCase):
    def test_source_swap_trains_identity_without_penalizing_shared_color(self):
        logits = torch.tensor([.3, -.4, .7], requires_grad=True)
        calls = []
        def sequence(prompt, boxes, value):
            calls.append((prompt, value))
            index = {'skirt': 0, 'shirt': 1, 'yellow': 2}[value]
            return torch.nn.functional.logsigmoid(logits[index]), [3, 7, 2][index]
        backend = SimpleNamespace(image=SimpleNamespace(width=10, height=10), sequence_logp=sequence)
        row = dict(objective='correction_pair', kind='binding', query=None, boxes=[[1, 1, 9, 9]],
                   positive='skirt', negative='shirt')
        metrics = backward_codex_objective(backend, row)
        self.assertAlmostEqual(metrics['positive'], math.log1p(math.exp(-.3)), places=6)
        self.assertAlmostEqual(metrics['negative'], math.log1p(math.exp(-.4)), places=6)
        self.assertLess(logits.grad[0].item(), 0)
        self.assertGreater(logits.grad[1].item(), 0)
        self.assertEqual(logits.grad[2].item(), 0)
        self.assertEqual(calls[0][0], calls[1][0])
        self.assertNotRegex(calls[0][0], r'\b(skirt|shirt|yellow)\b')

    def test_positive_replay_never_invents_a_negative(self):
        weight = torch.tensor(.2, requires_grad=True)
        calls = []
        def sequence(prompt, boxes, value):
            calls.append(value)
            return torch.nn.functional.logsigmoid(weight), 5
        backend = SimpleNamespace(image=SimpleNamespace(width=10, height=10), sequence_logp=sequence)
        row = dict(objective='positive_replay', query=dict(subject='bag', predicate='color'),
                   answer='red', boxes=[[0, 0, 8, 8]])
        result = backward_codex_objective(backend, row)
        self.assertEqual(calls, ['red'])
        self.assertEqual(result['negative_tokens'], 0)
        self.assertLess(weight.grad.item(), 0)
