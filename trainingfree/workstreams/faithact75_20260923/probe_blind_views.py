"""Small, read-only prompt/view diagnostic for independent blind inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from trainingfree.prompts import task_question
from .independent_blind_reader import Qwen3BlindReader


def crop(image: Image.Image, box: list[float]) -> Image.Image:
    width, height = image.size
    return image.crop((round(box[0] * width), round(box[1] * height),
                       round(box[2] * width), round(box[3] * height)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id", default="0")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh diagnostic output file")
    cases = {str(row["id"]): row for row in map(json.loads,
        args.dataset.read_text(encoding="utf-8").splitlines())}
    traces = {str(row["id"]): row for row in map(json.loads,
        args.trace.read_text(encoding="utf-8").splitlines())}
    case, trace = cases[args.id], traces[args.id]
    step = next(event["candidate"] for event in trace["trace"]
                if event.get("candidate", {}).get("kind") == "visual")
    box = next((event.get("audit", {}).get("scores", {}).get("blind_box")
                for event in trace["trace"] if event.get("audit")), None)
    image = Image.open(args.image_root / case["image"]).convert("RGB")
    reader = Qwen3BlindReader(args.model, "cuda:0")
    neutral = (
        "Inspect the image afresh and answer ONLY the neutral inspection question. "
        "No candidate claim or answer is available. Report one directly visible cue, "
        "not a guessed conclusion. If ambiguous, set resolved=false. "
        "Return JSON with observation and resolved.\n"
        f"Inspection question: {step['evidence_question']}"
    )
    variants = {
        "full_neutral": (image, neutral),
        "full_with_task_options": (
            image, neutral + "\nTask context, including possible values but no proposed answer:\n"
            + task_question(case["question"])),
        "full_plus_crop_neutral": (
            [image, crop(image, box)] if box else image,
            neutral + "\nImage 1 is the full scene; Image 2 is a magnified inspection region."
            if box else neutral),
    }
    if box:
        variants["crop_only_neutral"] = (crop(image, box), neutral)
    output = []
    for name, (view, prompt) in variants.items():
        value = json.loads(reader.generate(view, prompt))
        record = {"variant": name, "observation": value["observation"],
                  "resolved": value["resolved"], "raw": reader.last_raw}
        output.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"id": args.id, "reference": case["reference"],
                                       "box": box, "results": output}, ensure_ascii=False,
                                      indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
