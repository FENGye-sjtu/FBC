"""Reference-blind option-likelihood diagnostic for a frozen VLM."""

import argparse
import json
from pathlib import Path
import re
import time

from PIL import Image

from trainingfree.qwen import Qwen
from trainingfree.prompts import task_question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--limit', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--quantization', choices=['nf4', 'none'], default='nf4')
    parser.add_argument('--max-pixels', type=int, default=1003520)
    args = parser.parse_args()

    import torch
    torch.set_num_threads(4)
    torch.manual_seed(0)
    vlm = Qwen(args.model, args.device, args.max_pixels, 64, args.quantization)
    cases = [json.loads(line) for line in args.dataset.read_text(encoding='utf-8').splitlines()
             if line.strip()]
    cases = cases[args.offset:args.offset + args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        done = {str(json.loads(line)['id']) for line in args.output.read_text(encoding='utf-8').splitlines()
                if line.strip()}
    with args.output.open('a', encoding='utf-8') as stream:
        for case in cases:
            if str(case['id']) in done:
                continue
            labels = re.findall(r'(?m)^\s*([A-E])[.)]\s', case['question'])
            if len(labels) < 2:
                raise ValueError(f'Case {case["id"]} is not a multiple-choice question')
            image_path = Path(case['image'])
            if not image_path.is_absolute():
                image_path = args.image_root / image_path
            image = Image.open(image_path).convert('RGB')
            prompt = ('Independently choose the single best option from the image. '
                      'Return only the option letter.\nQuestion: ' + task_question(case['question']))
            started = time.monotonic()
            scores = vlm.option_log_probabilities(image, prompt, labels)
            answer = max(scores, key=scores.get)
            row = {'id': str(case['id']), 'answer': answer, 'scores': scores,
                   'correct': answer == case['reference'], 'reference': case['reference'],
                   'elapsed_seconds': time.monotonic() - started,
                   'method': 'frozen_option_log_probability'}
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            print(json.dumps({'id': row['id'], 'answer': answer, 'correct': row['correct']}),
                  flush=True)


if __name__ == '__main__':
    main()
