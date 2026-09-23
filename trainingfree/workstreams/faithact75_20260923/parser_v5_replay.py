"""Replay a parser change on immutable model outputs, never on references."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def rows(path: Path) -> dict[str, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = str(row["id"])
        if case_id in result:
            raise ValueError(f"duplicate ID {case_id} in {path}")
        result[case_id] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old, new = module(args.old, "old_answers"), module(args.new, "new_answers")
    cases = rows(args.cases)
    report = {"old_version": old.PARSER_VERSION, "new_version": new.PARSER_VERSION,
              "old_sha256": hashlib.sha256(args.old.read_bytes()).hexdigest(),
              "new_sha256": hashlib.sha256(args.new.read_bytes()).hexdigest(),
              "reference_blind_extraction": True, "runs": []}
    for path in args.run:
        previous_correct = current_correct = 0
        changed, valid_changed = [], []
        source = rows(path)
        for case_id, row in source.items():
            case = cases[case_id]
            raw = old.raw_answer(row)
            question, reference = case["question"], case["reference"]
            before = old.extract_answer(raw, question)
            after = new.extract_answer(raw, question)
            choices = old.choices_from_question(question)
            if before != after:
                entry = {"id": case_id, "old": before, "new": after,
                         "reference": reference, "old_correct": before == reference,
                         "new_correct": after == reference}
                changed.append(entry)
                if before in choices:
                    valid_changed.append(entry)
            previous_correct += before == reference
            current_correct += after == reference
        report["runs"].append({
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n": len(source), "old_correct": previous_correct,
            "new_correct": current_correct,
            "changed": changed, "already_valid_changed": valid_changed,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output), "old_version": old.PARSER_VERSION,
                      "new_version": new.PARSER_VERSION,
                      "runs": [{key: value for key, value in run.items()
                                if key in ("path", "n", "old_correct", "new_correct")}
                               | {"changed": len(run["changed"]),
                                  "already_valid_changed": len(run["already_valid_changed"])}
                               for run in report["runs"]]}, indent=2))


if __name__ == "__main__":
    main()
