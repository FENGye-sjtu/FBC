"""Cross-model blind observation with conservative semantic agreement gate."""

from __future__ import annotations

from trainingfree.core import Audit

from .evidence_first_v26 import EvidenceFirstLayoutAnchor
from .evidence_first_v27 import EvidenceFirstAdaptiveBlindGate, exact_blind_match


class EvidenceFirstCrossModelBlindGate(EvidenceFirstAdaptiveBlindGate):
    protocol_version = "evidence_first_v16_qwen3_candidate_blind_reader"
    relaxed_visual_support = 0.5
    independent_consistency = 0.65

    def audit(self, image, question, step, ledger):
        if step.kind != "visual":
            return EvidenceFirstLayoutAnchor.audit(self, image, question, step, ledger)
        normal_support = self.support
        try:
            self.support = min(normal_support, self.relaxed_visual_support)
            result = EvidenceFirstLayoutAnchor.audit(self, image, question, step, ledger)
        finally:
            self.support = normal_support
        scores = result.scores
        scores["blind_reader"] = getattr(self.vlm, "independent_reader_model", "unavailable")
        blind = scores.get("blind_observation") or {}
        if not isinstance(blind, dict):
            blind = {}
        observation = blind.get("observation", "")
        scores["exact_blind_match"] = bool(blind.get("resolved")) and exact_blind_match(
            step.claim, observation)
        if result.status != "supported":
            return result
        support, consistency = scores.get("support"), scores.get("blind_consistency")
        if not all(isinstance(value, (int, float)) for value in (support, consistency)):
            return Audit("inconclusive", scores, "Missing independent audit score")
        if support >= normal_support and consistency >= normal_support:
            scores["visual_gate"] = "ordinary_with_independent_reader"
            return result
        if (support >= self.relaxed_visual_support
                and consistency >= self.independent_consistency):
            scores["visual_gate"] = "independent_semantic_majority"
            return result
        if observation:
            self.last_blind_candidate = {
                "observation": observation, "box": scores.get("blind_box") or []}
        return Audit("inconclusive", scores,
                     "Independent visual reread is not sufficiently consistent; repair this point.")
