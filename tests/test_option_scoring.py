"""No-model checks of shared-prefix option scoring and its caller.

The small tensor double executes the real adapter methods with deterministic
causal logits. It checks indexing/normalization without requiring local torch.
Real GPU numerical agreement remains a separate integration measurement.
"""
from contextlib import nullcontext
import math
from types import SimpleNamespace
import unittest

from trainingfree.core import Step
from trainingfree.online import Online
from trainingfree.qwen import Qwen


class Tensor:
    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        shape, value = [], self.data
        while isinstance(value, list):
            shape.append(len(value))
            value = value[0] if value else None
        return tuple(shape)

    def __getitem__(self, key):
        def select(value, keys):
            if not keys:
                return value
            head, *tail = keys
            return ([select(v, tail) for v in value[head]] if isinstance(head, slice)
                    else select(value[head], tail))
        return Tensor(select(self.data, key if isinstance(key, tuple) else (key,)))

    def to(self, device):
        return self

    def log_softmax(self, axis):
        assert axis == -1
        def normalize(value):
            if isinstance(value[0], list):
                return [normalize(v) for v in value]
            pivot = max(value)
            total = pivot + math.log(sum(math.exp(v - pivot) for v in value))
            return [v - total for v in value]
        return Tensor(normalize(self.data))

    def unsqueeze(self, axis):
        assert axis == -1
        return Tensor([[[v] for v in row] for row in self.data])

    def gather(self, axis, indices):
        assert axis == -1
        return Tensor([[[vector[index[0]]] for vector, index in zip(row, index_row)]
                       for row, index_row in zip(self.data, indices.data)])

    def sum(self):
        def total(value):
            return sum(total(v) for v in value) if isinstance(value, list) else value
        return Tensor(total(self.data))

    def item(self):
        return self.data


class Batch(dict):
    __getattr__ = dict.__getitem__


class Tokenizer:
    TOKENS = {'A': [1], 'B': [2], 'C': [3], 'AA': [1, 1]}

    def encode(self, value, add_special_tokens=False):
        return list(self.TOKENS[value])

    def __call__(self, value, **kwargs):
        return SimpleNamespace(input_ids=Tensor([self.encode(value)]))


def fake_qwen():
    adapter = Qwen.__new__(Qwen)
    adapter.device = 'cpu'
    adapter.processor = SimpleNamespace(tokenizer=Tokenizer())
    adapter.calls = adapter.input_tokens = adapter.generated_tokens = 0
    adapter.forward_lengths = []
    adapter.torch = SimpleNamespace(
        inference_mode=nullcontext,
        cat=lambda tensors, dim: Tensor([sum((t.data[0] for t in tensors), [])]),
        ones_like=lambda tensor: Tensor([[1] * len(row) for row in tensor.data]),
    )
    adapter.inputs = lambda image, prompt: Batch(input_ids=Tensor([[4, 2, 4]]),
                                               attention_mask=Tensor([[1, 1, 1]]))

    def last_logits(inputs, count):
        ids = inputs.input_ids.data[0]
        adapter.forward_lengths.append(len(ids))
        # Each position depends only on its prefix, as in a causal decoder.
        logits = [[.07 * sum(ids[:position + 1]) * token - .2 * token * token
                   for token in range(6)] for position in range(len(ids))]
        return Tensor([logits[-count:]])
    adapter._last_logits = last_logits
    return adapter


class OptionScoringTests(unittest.TestCase):
    def test_shared_single_token_scores_equal_original_teacher_forcing(self):
        adapter = fake_qwen()
        labels = ['A', 'B', 'C']
        legacy = {label: adapter.log_probability(None, 'prompt', label) for label in labels}
        legacy_calls = adapter.calls
        legacy_tokens = adapter.input_tokens
        optimized = adapter.option_log_probabilities(None, 'prompt', labels)
        for label in labels:
            self.assertAlmostEqual(optimized[label], legacy[label], places=14)
        self.assertEqual(adapter.forward_lengths, [4, 4, 4, 3])
        self.assertEqual(adapter.calls - legacy_calls, 1)
        self.assertEqual(adapter.input_tokens - legacy_tokens, 3)
        # Preserve full-vocabulary probabilities, not option-only normalization.
        self.assertLess(sum(math.exp(score) for score in optimized.values()), 1.)

    def test_multitoken_choice_uses_original_complete_sequence_score(self):
        adapter = fake_qwen()
        expected = {label: adapter.log_probability(None, 'prompt', label) for label in ['A', 'AA']}
        adapter.forward_lengths.clear()
        before = adapter.calls
        actual = adapter.option_log_probabilities(None, 'prompt', ['A', 'AA'])
        self.assertEqual(actual, expected)
        self.assertEqual(adapter.calls - before, 2)
        self.assertEqual(adapter.forward_lengths, [4, 5])

    def test_online_prefers_shared_scorer_and_preserves_withholding_behavior(self):
        class VLM:
            def __init__(self):
                self.calls = []

            def log_probability(self, *args):
                raise AssertionError('The per-label fallback must not run')

            def option_log_probabilities(self, image, prompt, labels):
                self.calls.append((image, prompt, labels))
                return {'A': -3., 'B': -1.}

        vlm, image = VLM(), object()
        backend = Online(vlm)
        scores = {}
        audit = backend.option_counterfactual(
            image, 'Question: Which?\nA. One\nB. Two',
            Step('A proposed observation.', final_answer='A'), scores)
        self.assertEqual(len(vlm.calls), 1)
        self.assertIs(vlm.calls[0][0], image)
        self.assertEqual(vlm.calls[0][2], ['A', 'B'])
        self.assertEqual(audit.status, 'supported')
        self.assertTrue(scores['answer_withheld'])
        self.assertEqual(scores['option_counterfactual_best'], 'B')

    def test_online_legacy_backend_falls_back_per_label(self):
        class VLM:
            def __init__(self):
                self.calls = []

            def log_probability(self, image, prompt, completion):
                self.calls.append(completion)
                return {'A': -3., 'B': -1.}[completion]

        vlm, scores = VLM(), {}
        result = Online(vlm).option_counterfactual(
            None, 'Question: Which?\nA. One\nB. Two',
            Step('A proposed observation.', final_answer='B'), scores)
        self.assertIsNone(result)
        self.assertEqual(vlm.calls, ['A', 'B'])
        self.assertEqual(scores['option_counterfactual_scores'], {'A': -3., 'B': -1.})


if __name__ == '__main__':
    unittest.main()
