"""Run the answer-only-anchor stepwise controller through the shared evaluator."""

import trainingfree.evidence_first as evidence_first
from trainingfree.run import main

from .evidence_first_v32 import EvidenceFirstDirectAnchor


if __name__ == '__main__':
    evidence_first.EvidenceFirstOnlineRepair = EvidenceFirstDirectAnchor
    main()
