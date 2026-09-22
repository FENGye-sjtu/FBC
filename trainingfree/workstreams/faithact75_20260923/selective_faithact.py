"""Reference-blind gate for accepting a FaithAct answer revision.

The frozen v1 rule accepts a changed, valid MCQ label only when every declared
independent support run predicts the same revised label. References are used
after selection solely to report accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VALID_LABELS = frozenset("ABCDE")


def load(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = str(row["id"])
        if case_id in rows:
            raise ValueError(f"duplicate id {case_id} in {path}")
        rows[case_id] = row
    return rows


def valid_label(value: object) -> bool:
    return isinstance(value, str) and value in VALID_LABELS


def choose(anchor: dict, revision: dict, supports: list[dict]) -> tuple[dict, str]:
    anchor_answer = anchor.get("answer", "")
    revision_answer = revision.get("answer", "")
    if revision_answer == anchor_answer:
        return anchor, "agreement_keep_anchor"
    if not valid_label(revision_answer):
        return anchor, "invalid_revision_keep_anchor"
    if not supports or any(row.get("answer", "") != revision_answer for row in supports):
        return anchor, "uncorroborated_revision_keep_anchor"
    return revision, "all_supports_corroborate_revision"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--revision", required=True, type=Path)
    parser.add_argument("--support", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    anchor = load(args.anchor)
    revision = load(args.revision)
    support_sets = [load(path) for path in args.support]
    ids = set(anchor)
    if set(revision) != ids:
        raise ValueError("revision IDs differ from anchor IDs")
    for index, rows in enumerate(support_sets):
        missing = ids - set(rows)
        if missing:
            raise ValueError(f"support[{index}] is missing {len(missing)} anchor IDs")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    correct = 0
    changed = 0
    corrections = 0
    regressions = 0
    with args.output.open("x", encoding="utf-8") as stream:
        for case_id in sorted(ids, key=lambda value: int(value)):
            selected, reason = choose(anchor[case_id], revision[case_id],
                                      [rows[case_id] for rows in support_sets])
            answer = selected.get("answer", "")
            reference = anchor[case_id].get("reference", anchor[case_id].get("answer", ""))
            is_correct = bool(answer) and str(answer).strip().lower() == str(reference).strip().lower()
            anchor_correct = bool(anchor[case_id].get("correct"))
            selected_revision = selected is revision[case_id]
            changed += selected_revision
            corrections += selected_revision and is_correct and not anchor_correct
            regressions += selected_revision and anchor_correct and not is_correct
            correct += is_correct
            counts[reason] = counts.get(reason, 0) + 1
            stream.write(json.dumps({
                "id": case_id,
                "answer": answer,
                "reference": reference,
                "correct": is_correct,
                "anchor_answer": anchor[case_id].get("answer", ""),
                "revision_answer": revision[case_id].get("answer", ""),
                "support_answers": [rows[case_id].get("answer", "") for rows in support_sets],
                "selected": "revision" if selected_revision else "anchor",
                "decision": reason,
            }, ensure_ascii=False) + "\n")

    summary = {
        "protocol": "selective_faithact_v1_all_supports",
        "reference_blind_selection": True,
        "n": len(ids),
        "correct": correct,
        "accuracy": correct / len(ids),
        "accepted_revisions": changed,
        "corrections": corrections,
        "regressions": regressions,
        "decision_counts": counts,
        "inputs": {
            "anchor": str(args.anchor),
            "revision": str(args.revision),
            "supports": [str(path) for path in args.support],
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
