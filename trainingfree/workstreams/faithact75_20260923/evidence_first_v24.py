"""V24 promotes a blind reread into a new candidate, never directly into evidence."""

from __future__ import annotations

import re

from trainingfree.core import Step

from .evidence_first_v23 import EvidenceFirstBlindReadSemantic


def atomic_observation(text: str) -> str:
    """Keep the directly visible first clause and drop explanatory spillover."""
    value = str(text or "").strip()
    value = re.split(r",\s*(?:indicating|suggesting|implying|which\s+means)\b", value,
                     maxsplit=1, flags=re.I)[0]
    value = re.split(r"(?<=[.!?])\s+", value, maxsplit=1)[0]
    return value.rstrip(" ,;:") + ("" if value.endswith((".", "!", "?")) else ".")


class EvidenceFirstBlindRepair(EvidenceFirstBlindReadSemantic):
    protocol_version = "evidence_first_v11_blind_read_repair"

    def _start_case(self, image, question):
        self.last_blind_candidate = None
        return super()._start_case(image, question)

    def propose(self, image, question, ledger, feedback):
        pending = getattr(self, "last_blind_candidate", None)
        if feedback and pending and self.last_candidate is not None and self.last_candidate.kind == "visual":
            self.repair_round += 1
            claim = atomic_observation(pending["observation"])
            if not claim or claim == ".":
                self.last_blind_candidate = None
                return super().propose(image, question, ledger, feedback)
            step = Step(
                claim=claim,
                kind="visual",
                evidence_question=self.last_candidate.evidence_question,
                box=tuple(pending.get("box") or ()),
                final_answer=None,
            )
            self._reject_repeated_visual_claim(step, ledger)
            self.last_blind_candidate = None
            self.last_candidate = step
            self.pending_readback = None
            return step
        return super().propose(image, question, ledger, feedback)

    def audit(self, image, question, step, ledger):
        result = super().audit(image, question, step, ledger)
        blind = result.scores.get("blind_observation") if step.kind == "visual" else None
        if result.status != "supported" and isinstance(blind, dict) and blind.get("resolved"):
            observation = blind.get("observation")
            if isinstance(observation, str) and observation.strip():
                self.last_blind_candidate = {
                    "observation": observation,
                    "box": result.scores.get("blind_box") or [],
                }
        elif result.status == "supported":
            self.last_blind_candidate = None
        return result

