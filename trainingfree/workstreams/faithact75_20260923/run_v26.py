"""Run the isolated layout-anchored stepwise controller."""

import trainingfree.evidence_first as evidence_first

from .evidence_first_v26 import EvidenceFirstLayoutAnchor


evidence_first.EvidenceFirstOnlineRepair = EvidenceFirstLayoutAnchor

from trainingfree.run import main  # noqa: E402


if __name__ == "__main__":
    main()
