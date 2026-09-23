"""Exploratory, reference-blind arbitration of disagreeing answer anchors.

The sequential controller must independently return the proposed alternate
label with a verified status.  This is not a substitute for a paired held-out
evaluation of that controller and is not part of the frozen v1 selector.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

try:
    from .selective_faithact import load, valid_label
except ImportError:
    from selective_faithact import load, valid_label


def choose(anchor: dict, alternate: dict, controller: dict) -> tuple[str, str]:
    a, b, c = (row.get("answer", "") for row in (anchor, alternate, controller))
    if (valid_label(a) and valid_label(b) and a != b and c == b
            and controller.get("status") == "verified_by_model"):
        return b, "verified_alternate"
    return a, "keep_anchor"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    anchor, alternate, controller = (load(path) for path in
                                     (args.anchor, args.alternate, args.controller))
    if not set(anchor) <= set(alternate) & set(controller):
        raise ValueError("alternate/controller must include every anchor case")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with args.output.open("x", encoding="utf-8") as stream:
        for case_id in sorted(anchor, key=int):
            row = anchor[case_id]
            answer, decision = choose(row, alternate[case_id], controller[case_id])
            reference = row["reference"]
            counts["n"] += 1
            counts["correct"] += answer == reference
            counts["anchor_correct"] += row.get("answer") == reference
            counts["changes"] += decision == "verified_alternate"
            counts["corrections"] += decision == "verified_alternate" and answer == reference
            counts["regressions"] += decision == "verified_alternate" and row.get("answer") == reference
            stream.write(json.dumps({
                "id": case_id, "answer": answer, "reference": reference,
                "correct": answer == reference, "decision": decision,
                "anchor_answer": row.get("answer"),
                "alternate_answer": alternate[case_id].get("answer"),
                "controller_answer": controller[case_id].get("answer"),
                "controller_status": controller[case_id].get("status"),
            }, ensure_ascii=False) + "\n")
    report = {"protocol": "exploratory_disagreement_gate_v1",
              "reference_blind_selection": True, **dict(counts),
              "inputs": {"anchor": str(args.anchor), "alternate": str(args.alternate),
                         "controller": str(args.controller)}}
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
