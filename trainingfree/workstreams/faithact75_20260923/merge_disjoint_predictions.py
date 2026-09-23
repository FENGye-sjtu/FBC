"""Merge disjoint, resumable benchmark shards without changing any answer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def merge(paths: list[Path], expected_ids: list[str]) -> list[dict]:
    records: dict[str, dict] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["id"])
            if key in records:
                raise ValueError(f"Duplicate prediction ID: {key}")
            records[key] = row
    if set(records) != set(expected_ids):
        raise ValueError(f"Prediction IDs differ from dataset: missing={len(set(expected_ids)-set(records))}, "
                         f"extra={len(set(records)-set(expected_ids))}")
    return [records[key] for key in expected_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh merged output path")
    dataset = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    ids = [str(row["id"]) for row in dataset]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset has duplicate IDs")
    rows = merge(args.input, ids)
    if any(str(row.get("reference")) != str(case["reference"])
           for row, case in zip(rows, dataset)):
        raise ValueError("Reference mismatch between prediction and public dataset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                           encoding="utf-8")
    summary = {"n": len(rows), "correct": sum(row["correct"] for row in rows),
               "accuracy": sum(row["correct"] for row in rows) / len(rows),
               "inputs": [{"path": str(path),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for path in args.input],
               "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest()}
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
