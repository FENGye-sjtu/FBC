import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

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
from trainingfree.workstreams.faithact75_20260923.evidence_first_v27 import (
    EvidenceFirstAdaptiveBlindGate, exact_blind_match,
)
from trainingfree.workstreams.faithact75_20260923.evidence_first_v28 import (
    EvidenceFirstBlindMajorityGate,
)
from trainingfree.workstreams.faithact75_20260923.select_option_margin import select as select_margin
from trainingfree.workstreams.faithact75_20260923.evidence_first_v32 import EvidenceFirstDirectAnchor
from trainingfree.workstreams.faithact75_20260923.evidence_first_v29 import (
    EvidenceFirstCrossModelBlindGate,
)
from trainingfree.workstreams.faithact75_20260923.independent_blind_reader import (
    BLIND_MARKER, QwenWithIndependentBlindReader,
)
from trainingfree.qwen import Qwen
from trainingfree.evidence_first import EvidenceFirstOnlineRepair
from trainingfree.online_repair import OnlineRepair
from trainingfree.online import Online
from trainingfree.workstreams.faithact75_20260923.merge_disjoint_predictions import merge


class SelectiveFaithActTest(unittest.TestCase):
    def test_v32_initial_anchor_is_answer_only_not_upfront_chain(self):
        class VLM:
            def __init__(self):
                self.prompts = []
            def generate(self, image, prompt, **kwargs):
                self.prompts.append(prompt)
                return 'B'
        vlm = VLM()
        backend = EvidenceFirstDirectAnchor(vlm)
        backend._start_case(None, 'Question: Pick?\nA. left\nB. right')
        self.assertEqual(backend.anchor_answer, 'B')
        self.assertEqual(len(vlm.prompts), 1)
        self.assertTrue(vlm.prompts[0].endswith('Return only the concise answer.'))
        self.assertNotIn('step by step', vlm.prompts[0])
        self.assertEqual(backend.finalize(None, 'Question: Pick?\nA. left\nB. right', []), 'B')
        self.assertEqual(backend.last_finalization['source'], 'tentative_answer_only_anchor')

    def test_option_margin_selector_ignores_reference_for_decision(self):
        base = {'id': '0', 'answer': 'B', 'reference': 'C'}
        option = {'answer': 'C', 'scores': {'A': -2.0, 'B': -1.0, 'C': 0.0}}
        result = select_margin(base, option)
        self.assertTrue(result['changed'])
        self.assertEqual(result['answer'], 'C')
        self.assertEqual(select_margin({**base, 'reference': 'A'}, option)['answer'], 'C')
        self.assertFalse(select_margin(base, option, margin=1.1)['changed'])

    def test_disjoint_prediction_merge_rejects_overlap_and_missing_ids(self):
        with TemporaryDirectory() as folder:
            left, right = Path(folder) / "left.jsonl", Path(folder) / "right.jsonl"
            left.write_text('{"id":"0","answer":"A"}\n', encoding="utf-8")
            right.write_text('{"id":"1","answer":"B"}\n', encoding="utf-8")
            self.assertEqual([row["id"] for row in merge([left, right], ["0", "1"])],
                             ["0", "1"])
            with self.assertRaisesRegex(ValueError, "Duplicate prediction ID"):
                merge([left, left], ["0"])
            with self.assertRaisesRegex(ValueError, "Prediction IDs differ"):
                merge([left], ["0", "1"])

    def test_exact_blind_match_drops_explanatory_spillover_only(self):
        self.assertTrue(exact_blind_match(
            "The wheel points left.",
            "The wheel points left, indicating that the car may turn."))
        self.assertFalse(exact_blind_match(
            "The wheel points right.", "The wheel points left."))

    def test_v27_relaxes_only_exact_blind_visual_match(self):
        backend = object.__new__(EvidenceFirstAdaptiveBlindGate)
        backend.support = 0.8
        step = Step("The wheel points left.", evidence_question="Which way does the wheel point?")

        def fake_audit(instance, image, question, candidate, ledger):
            self.assertEqual(instance.support, 0.65)
            return Audit("supported", {
                "blind_observation": {"resolved": True,
                                      "observation": "The wheel points left."},
                "blind_consistency": 0.68, "support": 0.68,
            })

        with patch.object(EvidenceFirstLayoutAnchor, "audit", fake_audit):
            audit = backend.audit(None, "Question", step, [])
        self.assertEqual(audit.status, "supported")
        self.assertEqual(audit.scores["visual_gate"], "exact_blind_match_relaxed")
        self.assertEqual(backend.support, 0.8)

    def test_v28_majority_gate_still_requires_exact_blind_match(self):
        backend = object.__new__(EvidenceFirstBlindMajorityGate)
        backend.support = 0.8
        step = Step("The wheel points left.", evidence_question="Which way does the wheel point?")

        def fake_audit(instance, image, question, candidate, ledger):
            self.assertEqual(instance.support, 0.5)
            return Audit("supported", {
                "blind_observation": {"resolved": True,
                                      "observation": "The wheel points left."},
                "blind_consistency": 0.53, "support": 0.53,
            })

        with patch.object(EvidenceFirstLayoutAnchor, "audit", fake_audit):
            audit = backend.audit(None, "Question", step, [])
        self.assertEqual(audit.status, "supported")
        self.assertEqual(backend.support, 0.8)

    def test_v29_routes_only_neutral_blind_read_to_independent_model(self):
        backend = object.__new__(QwenWithIndependentBlindReader)
        backend.independent_reader_calls = 0
        backend.calls = backend.input_tokens = backend.generated_tokens = 0

        class FakeReader:
            input_tokens = 0
            generated_tokens = 0

            def generate(self, image, prompt, max_tokens):
                self.input_tokens += 10
                self.generated_tokens += 5
                return '{"observation":"right","resolved":true}'

        backend.independent_reader = FakeReader()
        with patch.object(Qwen, "generate", return_value="primary"):
            self.assertEqual(backend.generate(None, BLIND_MARKER + " question"),
                             '{"observation":"right","resolved":true}')
            self.assertEqual(backend.generate(None, "ordinary proposal"), "primary")
        self.assertEqual(backend.independent_reader_calls, 1)
        self.assertEqual((backend.calls, backend.input_tokens, backend.generated_tokens),
                         (1, 10, 5))

    def test_v29_requires_independent_semantic_agreement(self):
        backend = object.__new__(EvidenceFirstCrossModelBlindGate)
        backend.support = 0.8
        backend.vlm = type("Reader", (), {"independent_reader_model": "Qwen3"})()
        step = Step("The wheel points left.", evidence_question="Which way does the wheel point?")

        def fake_audit(instance, image, question, candidate, ledger):
            self.assertEqual(instance.support, 0.5)
            return Audit("supported", {
                "blind_observation": {"resolved": True,
                                      "observation": "The wheel points right."},
                "blind_consistency": 0.53, "support": 0.99,
            })

        with patch.object(EvidenceFirstLayoutAnchor, "audit", fake_audit):
            audit = backend.audit(None, "Question", step, [])
        self.assertEqual(audit.status, "inconclusive")
        self.assertEqual(backend.support, 0.8)

    def test_formatted_option_cannot_bypass_changed_answer_guard(self):
        backend = object.__new__(EvidenceFirstOnlineRepair)
        backend.counterfactual = False
        backend.anchor_answer = "B"
        question = "Question: Which way?\nA. Left\nB. Straight\nC. Right"
        ledger = [Step("The wheel points left.", kind="visual",
                       verification="verified")]
        deduction = Step("The wheel points left.", kind="deduction",
                         final_answer="A. Left")
        with patch.object(OnlineRepair, "audit", return_value=Audit("supported", {})):
            audit = backend.audit(None, question, deduction, ledger)
        self.assertEqual(audit.status, "supported")
        self.assertTrue(audit.scores["answer_withheld"])
        self.assertEqual(audit.scores["anchor_disagreement"]["proposed"], "A")

    def test_formatted_visual_answer_is_recorded_as_tentative_label(self):
        backend = object.__new__(OnlineRepair)
        backend.pending_readback = None
        backend.repair_style = "reread"
        backend.audited_candidates = []
        question = "Question: Which way?\nA. Left\nB. Right"
        step = Step("The wheel points left.", kind="visual",
                    final_answer="A. Left")
        with patch.object(Online, "audit", return_value=Audit("inconclusive", {"support": 0.6})):
            backend.audit(None, question, step, [])
        self.assertEqual(backend.audited_candidates[0]["answer"], "A")

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
        self.assertEqual(select_devices([(0, 18000), (1, 9000)], 16000, 1500,
                                        min_devices=2, min_each_free_mib=8500),
                         [(0, 18000), (1, 9000)])
        self.assertEqual(select_devices([(0, 18000), (1, 8000)], 16000, 1500,
                                        min_devices=2, min_each_free_mib=8500), [])

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
