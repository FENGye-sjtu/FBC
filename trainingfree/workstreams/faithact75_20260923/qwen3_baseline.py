"""Independent frozen Qwen3-VL control; never counted as Qwen2.5 method gain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from PIL import Image

from trainingfree.answers import extract_answer
from trainingfree.prompts import cot_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--quantization", choices=("nf4", "none"), default="nf4")
    args = parser.parse_args()
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": args.model, "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "offset": args.offset, "limit": args.limit, "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens, "quantization": args.quantization,
        "prompt": "evidence_layout_v5", "decoding": "greedy",
        "comparison_scope": "independent Qwen3 backbone, not Qwen2.5 method delta",
    }
    # Preserve the original low-resolution shard manifests so interrupted
    # controls remain resumable after adding the high-resolution variant.
    if args.min_pixels != 256 * 28 * 28:
        manifest["min_pixels"] = args.min_pixels
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing output has a different manifest")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=args.min_pixels,
                                              max_pixels=args.max_pixels)
    options = {"device_map": "cuda:0", "dtype": torch.bfloat16,
               "attn_implementation": "sdpa"}
    if args.quantization == "nf4":
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, **options).eval()
    model.requires_grad_(False)
    cases = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines()
             if line.strip()][args.offset:args.offset + args.limit]
    results = args.output / "cot.jsonl"
    done = ({str(json.loads(line)["id"])
             for line in results.read_text(encoding="utf-8").splitlines()}
            if results.exists() else set())
    with results.open("a", encoding="utf-8") as stream:
        for case in cases:
            if str(case["id"]) in done:
                continue
            start = time.monotonic()
            image = Image.open(args.image_root / case["image"]).convert("RGB")
            prompt = cot_prompt(case["question"], "evidence_layout_v5")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                        do_sample=False, use_cache=True)
            raw = processor.batch_decode(
                output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]
            answer = extract_answer(raw, case["question"])
            record = {"id": str(case["id"]), "raw": raw, "answer": answer,
                      "reference": case["reference"],
                      "correct": answer == case["reference"],
                      "elapsed_seconds": time.monotonic() - start,
                      "generated_tokens": output.shape[-1] - inputs.input_ids.shape[-1]}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({"id": record["id"], "correct": record["correct"],
                              "elapsed_seconds": record["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
