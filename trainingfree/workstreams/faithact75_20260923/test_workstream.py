import unittest

from PIL import Image

from trainingfree.core import Step
from trainingfree.workstreams.faithact75_20260923.isolated_bf16_queue import select_devices
from trainingfree.workstreams.faithact75_20260923.evidence_first_v22 import has_inference_language
from trainingfree.workstreams.faithact75_20260923.evidence_first_v23 import EvidenceFirstBlindReadSemantic
from trainingfree.workstreams.faithact75_20260923.evidence_first_v24 import atomic_observation
from trainingfree.workstreams.faithact75_20260923.evidence_first_v25 import observation_key
from trainingfree.workstreams.faithact75_20260923.selective_faithact import choose


class SelectiveFaithActTest(unittest.TestCase):
    def test_accepts_only_unanimously_supported_valid_revision(self):
        anchor = {"answer": "A"}
        revision = {"answer": "B"}
        self.assertIs(choose(anchor, revision, [{"answer": "B"}, {"answer": "B"}])[0], revision)
        self.assertIs(choose(anchor, revision, [{"answer": "B"}, {"answer": "A"}])[0], anchor)
        self.assertIs(choose(anchor, {"answer": "unknown"}, [{"answer": "unknown"}])[0], anchor)

    def test_queue_requires_declared_capacity_after_reserve(self):
        self.assertEqual(select_devices([(0, 10000), (1, 9000)], 16000, 1500),
                         [(0, 10000), (1, 9000)])
        self.assertEqual(select_devices([(0, 10000)], 9000, 1500), [])

    def test_visual_inference_language_is_rejected(self):
        self.assertTrue(has_inference_language("The road is uphill because it rises."))
        self.assertTrue(has_inference_language("The left position indicates the same direction."))
        self.assertFalse(has_inference_language("The lane edges converge above the image center."))

    def test_v23_defers_inference_check_to_semantic_audit(self):
        step = object()
        instance = object.__new__(EvidenceFirstBlindReadSemantic)
        self.assertIs(instance._validate_visual_language(step), step)

    def test_candidate_blind_audit_requires_independent_consistency(self):
        class FakeVLM:
            def __init__(self):
                self.prompts = []

            def generate(self, image, prompt, **kwargs):
                self.prompts.append(prompt)
                self.assertion = prompt
                return '{"observation":"The wheel visibly points right.","resolved":true}'

            def yes_probability(self, image, prompt):
                self.prompts.append(prompt)
                return 0.95

        backend = object.__new__(EvidenceFirstBlindReadSemantic)
        backend.vlm = FakeVLM()
        backend.support = 0.8
        backend.counterfactual = False
        backend.pending_readback = None
        backend.repair_style = "atomic_v1"
        backend.audited_candidates = []
        backend.anchor_answer = ""
        backend.locate = lambda image, query: (0.1, 0.1, 0.9, 0.9)
        image = Image.new("RGB", (20, 20), "white")
        step = Step("The wheel visibly points right.", evidence_question="Which way does the wheel point?")
        audit = backend.audit(image, "Question: Which way?", step, [])
        self.assertEqual(audit.status, "supported")
        self.assertNotIn(step.claim, backend.vlm.prompts[0])
        self.assertEqual(audit.scores["blind_consistency"], 0.95)

    def test_blind_repair_keeps_visible_clause_not_explanation(self):
        value = atomic_observation(
            "The wheel visibly points left, indicating that the car may turn left"
        )
        self.assertEqual(value, "The wheel visibly points left.")

    def test_blind_repair_deduplicates_image_suffix(self):
        self.assertEqual(
            observation_key("The wheel faces forward in the image."),
            observation_key("The wheel faces forward."),
        )


if __name__ == "__main__":
    unittest.main()
