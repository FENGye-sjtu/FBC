"""V23 keeps candidate-blind rereading without V22's lexical rejection gate."""

from .evidence_first_v22 import EvidenceFirstBlindRead


class EvidenceFirstBlindReadSemantic(EvidenceFirstBlindRead):
    protocol_version = "evidence_first_v10_candidate_blind_semantic_audit"

    def _validate_visual_language(self, step):
        # V22 showed that correct rereads often contain connective words such as
        # "indicating". The candidate-blind entailment audit, rather than a
        # brittle word list, decides whether the claim adds unsupported content.
        return step

