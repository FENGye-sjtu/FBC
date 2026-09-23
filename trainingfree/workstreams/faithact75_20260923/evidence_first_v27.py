"""V27 repairs V26's blind-read deadlock without asserting correctness.

The ordinary 0.8 gate remains in force. A lower 0.65 visual gate is allowed
only when the independent candidate-blind observation is lexically the same
atomic fact as the proposed claim; the ordinary image-support gate must also
pass at 0.65. Deductions continue to require 0.8. These are model-support
heuristics, not calibrated probabilities or proof of visual correctness.
"""

from __future__ import annotations

from trainingfree.core import Audit

from .evidence_first_v24 import atomic_observation
from .evidence_first_v25 import observation_key
from .evidence_first_v26 import EvidenceFirstLayoutAnchor


def exact_blind_match(claim: str, observation: str) -> bool:
    return observation_key(claim) == observation_key(atomic_observation(observation))


class EvidenceFirstAdaptiveBlindGate(EvidenceFirstLayoutAnchor):
    protocol_version = "evidence_first_v14_exact_blind_visual_gate"
    relaxed_visual_support = 0.65

    def propose(self, image, question, ledger, feedback):
        step = super().propose(image, question, ledger, feedback)
        if step.kind == "visual" and not step.evidence_question.strip():
            raise ValueError("Visual evidence requires a neutral evidence_question")
        return step

    def audit(self, image, question, step, ledger):
        if step.kind != "visual":
            return super().audit(image, question, step, ledger)
        normal_support = self.support
        try:
            self.support = min(normal_support, self.relaxed_visual_support)
            result = super().audit(image, question, step, ledger)
        finally:
            self.support = normal_support
        blind = result.scores.get("blind_observation") or {}
        if not isinstance(blind, dict):
            blind = {}
        observation = blind.get("observation", "")
        matched = bool(blind.get("resolved")) and exact_blind_match(step.claim, observation)
        visual_support = result.scores.get("support")
        blind_score = result.scores.get("blind_consistency")
        result.scores["exact_blind_match"] = matched
        if result.status != "supported":
            return result
        if (isinstance(visual_support, (int, float)) and visual_support >= normal_support
                and isinstance(blind_score, (int, float)) and blind_score >= normal_support):
            result.scores["visual_gate"] = "ordinary"
            return result
        if (matched and isinstance(visual_support, (int, float))
                and visual_support >= self.relaxed_visual_support
                and isinstance(blind_score, (int, float))
                and blind_score >= self.relaxed_visual_support):
            result.scores["visual_gate"] = "exact_blind_match_relaxed"
            return result
        if observation:
            self.last_blind_candidate = {"observation": observation,
                                         "box": result.scores.get("blind_box") or []}
        return Audit("inconclusive", result.scores,
                     "Sub-0.8 visual scores lack exact candidate-blind agreement; repair this point.")
