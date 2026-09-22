"""Run the isolated V22 controller through the existing reproducible harness."""

import trainingfree.evidence_first as evidence_first

from .evidence_first_v22 import EvidenceFirstBlindRead


evidence_first.EvidenceFirstOnlineRepair = EvidenceFirstBlindRead

from trainingfree.run import main  # noqa: E402


if __name__ == "__main__":
    main()

