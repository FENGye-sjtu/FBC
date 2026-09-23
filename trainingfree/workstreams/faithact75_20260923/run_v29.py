"""Two-frozen-model candidate-blind V29 pilot with the shared task harness."""

import json
import os
from pathlib import Path
import sys

import trainingfree.evidence_first as evidence_first
import trainingfree.qwen as qwen
from trainingfree.run import main

from .evidence_first_v29 import EvidenceFirstCrossModelBlindGate
from .independent_blind_reader import QwenWithIndependentBlindReader


if __name__ == "__main__":
    if "--output" not in sys.argv:
        raise ValueError("V29 requires --output for independent reader provenance")
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    output.mkdir(parents=True, exist_ok=True)
    (output / "independent_reader.json").write_text(json.dumps({
        "model": os.environ["CVPR_SECONDARY_MODEL"],
        "role": "candidate-blind visual observation only",
        "precision": "NF4",
        "not_a_single_model_method": True,
    }, indent=2), encoding="utf-8")
    qwen.Qwen = QwenWithIndependentBlindReader
    evidence_first.EvidenceFirstOnlineRepair = EvidenceFirstCrossModelBlindGate
    main()
