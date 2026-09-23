"""Create new answer labels from immutable raw outputs with one parser version."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parser", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("selected_answers", args.parser)
    if spec is None or spec.loader is None:
        raise ValueError(args.parser)
    answers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(answers)
    cases = {str(row["id"]): row for line in args.cases.read_text(encoding="utf-8").splitlines()
             if (row := json.loads(line))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for line in args.run.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            source = json.loads(line)
            case = cases[str(source["id"])]
            answer = answers.extract_answer(answers.raw_answer(source), case["question"])
            stream.write(json.dumps({
                "id": str(source["id"]), "answer": answer,
                "reference": case["reference"],
                "correct": answer == case["reference"],
                "parser_version": answers.PARSER_VERSION,
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
