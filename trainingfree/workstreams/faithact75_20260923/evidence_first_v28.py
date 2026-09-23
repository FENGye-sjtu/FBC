"""Two-signal visual pilot: blind exact match plus yes/no support majority.

The 0.5 support gate is a *hypothesis* for engineering evaluation, not a
calibrated truth probability. Deductions retain their stricter 0.8 gate.
"""

from .evidence_first_v27 import EvidenceFirstAdaptiveBlindGate


class EvidenceFirstBlindMajorityGate(EvidenceFirstAdaptiveBlindGate):
    protocol_version = "evidence_first_v15_blind_exact_majority_visual_gate"
    relaxed_visual_support = 0.5
