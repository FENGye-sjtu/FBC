"""V25 avoids promoting a blind observation equivalent to the audited claim."""

from __future__ import annotations

import re

from .evidence_first_v24 import EvidenceFirstBlindRepair, atomic_observation


def observation_key(text: str) -> str:
    value = re.sub(r"\b(?:in|from)\s+the\s+image\b", "", str(text or ""), flags=re.I)
    return re.sub(r"\W+", " ", value.lower()).strip()


class EvidenceFirstBlindRepairDeduplicated(EvidenceFirstBlindRepair):
    protocol_version = "evidence_first_v12_blind_read_repair_deduplicated"

    def audit(self, image, question, step, ledger):
        result = super().audit(image, question, step, ledger)
        pending = getattr(self, "last_blind_candidate", None)
        if pending and observation_key(atomic_observation(pending["observation"])) == observation_key(step.claim):
            self.last_blind_candidate = None
        return result

