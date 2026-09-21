"""Shared dataset, decoding, trace, and exact-match accuracy harness."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from .core import solve
from .online import Online


def extract_answer(text, question=''):
    if text is None:
        return ''
    text = text.strip()
    found = re.findall(r'(?:FINAL ANSWER\s*:|Thus,|Therefore,\s*(?:the answer is)?)(.*)', text, flags=re.I)
    if found:
        text = found[-1].strip()
    text = text.strip(' *`\n.').strip()
    text = re.sub(r'^(?:the\s+)?(?:correct\s+)?answer\s+is\s*:?\s*', '', text, flags=re.I)
    text = text.strip(' *`\n.')
    letter = re.fullmatch(r'\(?([A-E])\)?[.)]?', text)
    if not letter and re.search(r'\bA[.)]\s', question) and re.search(r'\bB[.)]\s', question):
        # Accept a single explicit option label plus its wording, not a list of options.
        labels = re.findall(r'\b([A-E])[.)]\s', text)
        if len(labels) == 1 and re.match(r'^[A-E][.)]\s', text):
            return labels[0]
    return letter.group(1) if letter else text


def normalize(text):
    return re.sub(r'\s+', ' ', str(text or '').strip().lower()).strip(' .!')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--image-root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--methods', default='cot,online,online_cf')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--max-pixels', type=int, default=1003520)
    p.add_argument('--max-steps', type=int, default=6)
    p.add_argument('--repairs', type=int, default=2)
    p.add_argument('--support', type=float, default=.8)
    p.add_argument('--specificity', type=float, default=.05)
    p.add_argument('--upstream', type=Path)
    p.add_argument('--clip')
    p.add_argument('--dino')
    p.add_argument('--poll-checkpoint')
    p.add_argument('--tool-device', default='cpu')
    args = p.parse_args()
    methods = args.methods.split(',')
    if not set(methods) <= {'direct', 'cot', 'faithact', 'online', 'online_cf'}:
        p.error('Unknown method')
    if 'faithact' in methods and not all([args.upstream, args.clip, args.dino, args.poll_checkpoint]):
        p.error('FaithAct requires upstream, clip, dino and poll-checkpoint')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest['dataset_sha256'] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    manifest['metric'] = 'strict_normalized_exact_match; unanswered/errors count incorrect'
    manifest['protocol'] = 'engineering_port_v1_not_yet_paper_matched'
    manifest['code_sha256'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in sorted(Path(__file__).parent.glob('*.py'))}
    # Never mix changed configurations into a resumed run.
    path = args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Existing run has a different manifest; choose a new output directory')
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    cases = [json.loads(line) for line in args.dataset.read_text(encoding='utf-8').splitlines() if line.strip()]
    cases = cases[args.offset:args.offset+args.limit if args.limit else None]
    if len({str(c['id']) for c in cases}) != len(cases):
        raise ValueError('Duplicate case IDs')
    from PIL import Image
    from .qwen import Qwen
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(0)
    vlm = Qwen(args.model, args.device, args.max_pixels, args.max_new_tokens)
    if 'faithact' in methods:
        from .faithact import FaithAct
        faithact = FaithAct(vlm, args.upstream, args.clip, args.dino, args.poll_checkpoint, args.tool_device)
    summary = {}
    for method in methods:
        result_path = args.output/(method+'.jsonl')
        prior = [json.loads(s) for s in result_path.read_text(encoding='utf-8').splitlines()] if result_path.exists() else []
        done = {str(r['id']) for r in prior}
        records = list(prior)
        backend = Online(vlm, method == 'online_cf', args.support, args.specificity)
        with result_path.open('a', encoding='utf-8') as stream:
            for case in cases:
                if str(case['id']) in done:
                    continue
                before = (vlm.calls, vlm.generated_tokens, vlm.input_tokens)
                start = time.monotonic()
                cached_cot_cost = {'elapsed_seconds': 0, 'calls': 0, 'generated_tokens': 0, 'input_tokens': 0}
                try:
                    image_path = Path(case['image'])
                    if not image_path.is_absolute():
                        image_path = (args.image_root or args.dataset.parent)/image_path
                    image = Image.open(image_path).convert('RGB')
                    question = case['question']
                    if method.startswith('online'):
                        record = solve(backend, image, question, args.max_steps, args.repairs)
                    elif method == 'direct':
                        record = {'raw': vlm.generate(image, question+'\nReturn only the concise answer.'), 'status': 'direct'}
                    else:
                        cot_prompt = question.replace('Please answer directly with only the letter of the correct option and nothing else.', '').replace('Please answer directly with a single word or number.', '')
                        cot_prompt += ('\nReason step by step. Start with a short introduction, separate numbered '
                                       'reasoning steps by blank lines, and finish with FINAL ANSWER: '
                                       'followed by only a letter, word, or number as required.')
                        if method == 'faithact':
                            cot_path = args.output/'cot.jsonl'
                            cots = {str(r['id']): r for r in map(json.loads, cot_path.read_text(encoding='utf-8').splitlines())} if cot_path.exists() else {}
                            cached = cots.get(str(case['id']), {})
                            cot = cached.get('raw') or vlm.generate(image, cot_prompt)
                            if cached.get('raw'):
                                cached_cot_cost = {key: cached.get(key, 0) for key in cached_cot_cost}
                            record = faithact.run(image, question, cot)
                            record['initial_cot_reused'] = bool(cached.get('raw'))
                        else:
                            record = {'raw': vlm.generate(image, cot_prompt), 'status': 'cot'}
                    answer = extract_answer(record.get('answer', record.get('raw')), question)
                except Exception as exc:
                    # Persist failures; never report accuracy only on successful cases.
                    record, answer = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}, ''
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                reference = case.get('reference', case.get('answer', ''))
                record.update(id=case['id'], method=method, answer=answer, reference=reference,
                              correct=bool(answer) and normalize(answer) == normalize(reference),
                              elapsed_seconds=time.monotonic()-start, calls=vlm.calls-before[0],
                              generated_tokens=vlm.generated_tokens-before[1], input_tokens=vlm.input_tokens-before[2])
                record['incremental_elapsed_seconds'] = record['elapsed_seconds']
                for key, value in cached_cot_cost.items():
                    record[key] += value
                stream.write(json.dumps(record, ensure_ascii=False)+'\n'); stream.flush()
                records.append(record)
                print(json.dumps({k: record[k] for k in ('id','method','status','answer','correct','elapsed_seconds')}), flush=True)
                if record['status'] == 'error':
                    # A broken adapter must be fixed before consuming an entire benchmark.
                    raise RuntimeError(record['error'])
        summary[method] = {'n': len(records), 'accuracy': sum(r['correct'] for r in records)/max(1,len(records)),
                           'answered': sum(bool(r['answer']) for r in records),
                           'mean_seconds': sum(r['elapsed_seconds'] for r in records)/max(1,len(records))}
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
