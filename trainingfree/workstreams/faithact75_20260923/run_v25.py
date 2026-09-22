"""Run the isolated V25 controller through the existing harness."""

import trainingfree.evidence_first as evidence_first

from .evidence_first_v25 import EvidenceFirstBlindRepairDeduplicated


evidence_first.EvidenceFirstOnlineRepair = EvidenceFirstBlindRepairDeduplicated

from trainingfree.run import main  # noqa: E402


if __name__ == "__main__":
    main()

