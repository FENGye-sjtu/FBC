"""Reference-blind majority answer anchor from already measured prompts.

This is a compute-matched diagnostic and a possible uncertain fallback for the
stepwise controller, not evidence verification or a paper-method result.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

try:
    from .selective_faithact import load, valid_label
except ImportError:  # standalone copy under the isolated server code directory
    from selective_faithact import load, valid_label


def choose(rows: list[dict], priority: int = 0) -> tuple[str, str]:
    labels = [row.get("answer", "") for row in rows]
    votes = Counter(value for value in labels if valid_label(value))
    if votes:
        top = votes.most_common()
        if top[0][1] >= 2 and (len(top) == 1 or top[0][1] > top[1][1]):
            return top[0][0], "majority"
    fallback = labels[priority]
    return (fallback if valid_label(fallback) else "", "priority_tie")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input) < 2 or not 0 <= args.priority < len(args.input):
        parser.error("need at least two inputs and a valid priority index")
    methods = [load(path) for path in args.input]
    ids = set(methods[0])
    if any(set(method) != ids for method in methods[1:]):
        parser.error("all inputs must contain the same case IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outcomes = Counter()
    decisions = Counter()
    with args.output.open("x", encoding="utf-8") as stream:
        for case_id in sorted(ids, key=int):
            members = [method[case_id] for method in methods]
            references = {row.get("reference") for row in members}
            if len(references) != 1:
                raise ValueError(f"reference mismatch for ID {case_id}")
            answer, decision = choose(members, args.priority)
            reference = next(iter(references))
            correct = answer == reference
            outcomes["correct"] += correct
            decisions[decision] += 1
            stream.write(json.dumps({
                "id": case_id, "answer": answer, "reference": reference,
                "correct": correct, "decision": decision,
                "input_answers": [row.get("answer", "") for row in members],
            }, ensure_ascii=False) + "\n")
    summary = {
        "protocol": "reference_blind_majority_v1",
        "n": len(ids), "correct": outcomes["correct"],
        "accuracy": outcomes["correct"] / len(ids),
        "decision_counts": dict(decisions),
        "inputs": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in args.input],
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
