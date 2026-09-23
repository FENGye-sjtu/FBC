"""Run the isolated V27 stepwise controller."""

import trainingfree.evidence_first as evidence_first

from .evidence_first_v27 import EvidenceFirstAdaptiveBlindGate


evidence_first.EvidenceFirstOnlineRepair = EvidenceFirstAdaptiveBlindGate

from trainingfree.run import main  # noqa: E402


if __name__ == "__main__":
    main()
