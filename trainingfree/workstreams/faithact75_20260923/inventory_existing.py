"""Read-only inventory and paired-oracle diagnostics for completed JSONL runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row["id"])
        if key in rows:
            raise ValueError(f"duplicate id {key} in {path}")
        rows[key] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    methods = {path.stem + "@" + path.parent.name: load(path) for path in args.paths}
    common = set.intersection(*(set(rows) for rows in methods.values()))
    if not common:
        raise ValueError("inputs have no common IDs")
    names = list(methods)
    report = {
        "inputs": {name: str(path) for name, path in zip(names, args.paths)},
        "common_ids": len(common),
        "scores": {
            name: sum(bool(methods[name][case_id].get("correct")) for case_id in common)
            for name in names
        },
        "pairs": {},
    }
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            left_only = [case_id for case_id in common if methods[left][case_id].get("correct")
                         and not methods[right][case_id].get("correct")]
            right_only = [case_id for case_id in common if methods[right][case_id].get("correct")
                          and not methods[left][case_id].get("correct")]
            both = [case_id for case_id in common if methods[left][case_id].get("correct")
                    and methods[right][case_id].get("correct")]
            report["pairs"][left + " <> " + right] = {
                "both_correct": len(both),
                "left_only": len(left_only),
                "right_only": len(right_only),
                "oracle_union": len(both) + len(left_only) + len(right_only),
                "left_only_ids": sorted(left_only, key=lambda value: int(value)),
                "right_only_ids": sorted(right_only, key=lambda value: int(value)),
            }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
