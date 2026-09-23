"""Evidence-First with an answer-only uncertain anchor, never an upfront chain."""

from trainingfree.answers import extract_answer
from trainingfree.evidence_first import EvidenceFirstOnlineRepair


class EvidenceFirstDirectAnchor(EvidenceFirstOnlineRepair):
    protocol_version = 'evidence_first_v32_answer_only_anchor'

    def _start_case(self, image, question):
        self.audited_candidates = []
        self.last_finalization = None
        self.pending_readback = None
        self.last_candidate = None
        self.repair_round = 0
        # Unlike earlier engineering variants, the tentative fallback does
        # not produce a reasoning chain before the first stepwise audit.
        self.anchor_raw = self.vlm.generate(image, question + '\nReturn only the concise answer.')
        self.anchor_answer = extract_answer(self.anchor_raw, question)
        self.fallback_answer = self.anchor_answer
