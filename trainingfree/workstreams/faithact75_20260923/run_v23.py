"""Run the isolated V23 controller through the existing harness."""

import trainingfree.evidence_first as evidence_first

from .evidence_first_v23 import EvidenceFirstBlindReadSemantic


evidence_first.EvidenceFirstOnlineRepair = EvidenceFirstBlindReadSemantic

from trainingfree.run import main  # noqa: E402


if __name__ == "__main__":
    main()

