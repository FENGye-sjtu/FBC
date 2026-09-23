"""V26 uses the stronger layout-v5 CoT answer as an uncertain fallback.

The layout chain is never inserted into the verified evidence ledger.  Visual
facts still pass the candidate-blind and image-support checks before the next
reasoning step, and repairs remain limited to three attempts.
"""

from trainingfree.answers import extract_answer
from trainingfree.prompts import cot_prompt

from .evidence_first_v25 import EvidenceFirstBlindRepairDeduplicated


class EvidenceFirstLayoutAnchor(EvidenceFirstBlindRepairDeduplicated):
    protocol_version = "evidence_first_v13_layout_anchor_blind_repair"

    def _start_case(self, image, question):
        self.audited_candidates = []
        self.last_finalization = None
        self.pending_readback = None
        self.last_candidate = None
        self.last_blind_candidate = None
        self.repair_round = 0
        self.anchor_raw = self.vlm.generate(
            image, cot_prompt(question, "evidence_layout_v5"))
        self.anchor_answer = extract_answer(self.anchor_raw, question)
        self.fallback_answer = self.anchor_answer
