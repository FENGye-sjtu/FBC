"""Use the v28 audit gate with the unchanged shared benchmark runner."""

import trainingfree.evidence_first as evidence_first
from trainingfree.run import main

from .evidence_first_v28 import EvidenceFirstBlindMajorityGate


if __name__ == "__main__":
    evidence_first.EvidenceFirstOnlineRepair = EvidenceFirstBlindMajorityGate
    main()
