"""Run the isolated V24 controller through the existing harness."""

import trainingfree.evidence_first as evidence_first

from .evidence_first_v24 import EvidenceFirstBlindRepair


evidence_first.EvidenceFirstOnlineRepair = EvidenceFirstBlindRepair

from trainingfree.run import main  # noqa: E402


if __name__ == "__main__":
    main()

