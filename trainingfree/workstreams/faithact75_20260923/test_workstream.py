import unittest

from PIL import Image

from trainingfree.core import Audit, Step, solve
from trainingfree.workstreams.faithact75_20260923.isolated_bf16_queue import select_devices
from trainingfree.workstreams.faithact75_20260923.evidence_first_v22 import has_inference_language
from trainingfree.workstreams.faithact75_20260923.evidence_first_v23 import EvidenceFirstBlindReadSemantic
from trainingfree.workstreams.faithact75_20260923.evidence_first_v24 import atomic_observation
from trainingfree.workstreams.faithact75_20260923.evidence_first_v25 import observation_key
from trainingfree.workstreams.faithact75_20260923.selective_faithact import choose
from trainingfree.workstreams.faithact75_20260923.ensemble_anchor import choose as vote
from trainingfree.workstreams.faithact75_20260923.evidence_first_v26 import EvidenceFirstLayoutAnchor
from trainingfree.workstreams.faithact75_20260923.disagreement_gate import choose as gate


class SelectiveFaithActTest(unittest.TestCase):
    def test_disagreement_gate_requires_verified_agreement(self):
        anchor, alternate = {"answer": "A"}, {"answer": "B"}
        self.assertEqual(gate(anchor, alternate,
                              {"answer": "B", "status": "verified_by_model"}),
                         ("B", "verified_alternate"))
        self.assertEqual(gate(anchor, alternate,
                              {"answer": "B", "status": "answered_with_uncertainty"}),
                         ("A", "keep_anchor"))

    def test_v26_anchor_chain_is_not_promoted_to_ledger(self):
        class FakeVLM:
            def generate(self, image, prompt, **kwargs):
                self.prompt = prompt
                return "Visible cue.\nFinal answer: B"

        backend = object.__new__(EvidenceFirstLayoutAnchor)
        backend.vlm = FakeVLM()
        backend._start_case(None, "Question: Which?\nOptions:\nA. left\nB. right")
        self.assertEqual(backend.anchor_answer, "B")
        self.assertEqual(backend.fallback_answer, "B")
        self.assertIn("Evidence:\n\n1. <visible fact>", backend.vlm.prompt)

    def test_majority_anchor_uses_reference_blind_labels(self):
        self.assertEqual(vote([{"answer": "B"}, {"answer": "A"}, {"answer": "B"}]),
                         ("B", "majority"))
        self.assertEqual(vote([{"answer": "B"}, {"answer": "A"}, {"answer": "C"}]),
                         ("B", "priority_tie"))

    def test_quoted_null_is_not_a_verified_answer(self):
        step = Step.from_dict({
            "claim": "The visible evidence is insufficient.",
            "kind": "deduction",
            "evidence_question": "",
            "final_answer": "null",
        })
        self.assertIsNone(step.final_answer)

        class FakeBackend:
            def propose(self, image, question, ledger, feedback):
                return step

            def audit(self, image, question, proposed, ledger):
                return Audit("supported")

            def finalize(self, image, question, ledger):
                return "A"

        result = solve(FakeBackend(), None, "Question", max_steps=1)
        self.assertEqual(result["status"], "answered_with_uncertainty")
        self.assertIsNone(result["ledger"][0]["final_answer"])

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
