"""Shared dataset, decoding, trace, and exact-match accuracy harness."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import shutil
import traceback
from collections import Counter

from .core import solve
from .online import Online
from .prompts import cot_prompt
from .answers import PARSER_VERSION, extract_answer, raw_answer


def choices_from_question(question):
    """Return declared MCQ choices without consulting the reference answer."""
    return {match.group(1).upper(): match.group(2).strip()
            for match in re.finditer(r'(?m)^\s*([A-E])[.)]\s+(.+?)\s*$', question or '')}


def _answer_text(text):
    return re.sub(r'[^a-z0-9]+', ' ', str(text).lower()).strip()


def legacy_extract_answer(text, question=''):
    if text is None:
        return ''
    text = text.strip()
    if not text:
        return ''
    choices = choices_from_question(question)
    has_choices = len(choices) >= 2 or (
        re.search(r'\bA[.)]\s', question or '') and re.search(r'\bB[.)]\s', question or ''))
    tail = text.splitlines()[-1].strip(' *`.')
    if re.fullmatch(r'[A-E]', tail) and has_choices:
        return tail
    if len(choices) >= 2:
        tail_choice = re.match(r'^\**\(?([A-E])\)?[.):]\**\s*(.+?)\s*$', tail)
        if tail_choice and _answer_text(tail_choice.group(2)) == _answer_text(
                choices.get(tail_choice.group(1), '')):
            return tail_choice.group(1)
    # Use the final explicit conclusion. These markers are model output only;
    # extraction never receives or consults the reference label.
    markers = list(re.finditer(
        r'(?:\b(?:final|correct|best|most\s+(?:reasonable|plausible))\s+'
        r'(?:answer|choice|estimate)\s*(?:is|would\s+be)?\s*:|\banswer\s+is\s*:|'
        r'\bThus,|\bTherefore,|\bGiven these (?:observations|points),)', text, flags=re.I))
    if markers:
        text = text[markers[-1].end():].strip()
    text = text.strip(' *`\n.').strip()
    text = re.sub(r'^(?:the\s+)?(?:(?:correct|final)\s+)?answer\s*(?:is)?\s*:?\s*', '', text, flags=re.I)
    text = text.strip(' *`\n.')
    letter = re.fullmatch(r'\(?([A-E])\)?[.)]?', text)
    if not letter and has_choices:
        # Accept an explicit option at the start of the conclusion even when a
        # short post-answer explanation follows (common in FaithAct outputs).
        leading = re.match(r'^\**\(?([A-E])\)?[.):]\**(?:\s|$)', text)
        labels = re.findall(r'(?<![A-Za-z])([A-E])[.):](?=\s|\*)', text)
        if leading and len(labels) == 1:
            return leading.group(1)
        # Some valid answers state the option wording rather than its label.
        # Map only a unique declared option phrase in the conclusion, avoiding
        # guesses when several choices are repeated or contrasted.
        if len(choices) >= 2:
            normalized = _answer_text(text)
            matched = [label for label, value in choices.items()
                       if _answer_text(value) and re.search(
                           r'(?<![a-z0-9])' + re.escape(_answer_text(value)) + r'(?![a-z0-9])', normalized)]
            if len(matched) == 1:
                return matched[0]
    return letter.group(1) if letter else text


def normalize(text):
    return re.sub(r'\s+', ' ', str(text or '').strip().lower()).strip(' .!')


def main(argv=None, shared=None):
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--image-root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--methods', default='cot,online,online_cf')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--quantization', choices=['none','nf4','awq'], default='none')
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--max-pixels', type=int, default=1003520)
    p.add_argument('--max-steps', type=int, default=6)
    p.add_argument('--cot-style', choices=['reasoned_v2','grounded_concise_v3','preserve_select_v3',
                                         'evidence_brief_v4','evidence_adaptive_v4','evidence_layout_v5',
                                         'discriminative_evidence_v6','discriminative_layout_v7',
                                         'faithact_gcot_official','faithact_paper_cot_literal'],
                   default='reasoned_v2')
    p.add_argument('--repairs', type=int, choices=range(4), default=3)
    p.add_argument('--initial-style', choices=['full','compact_v1'], default='full',
                   help='Only changes the first atomic proposal; all audits and repairs are unchanged')
    p.add_argument('--repair-style', choices=['reread','atomic_v1'], default='reread',
                   help='Optional concise repair observation/claim variant; no audit bypass')
    p.add_argument('--support', type=float, default=.8)
    p.add_argument('--specificity', type=float, default=.05)
    p.add_argument('--upstream', type=Path)
    p.add_argument('--clip')
    p.add_argument('--dino')
    p.add_argument('--poll-checkpoint')
    p.add_argument('--tool-device', default='cpu')
    p.add_argument('--faithact-box-threshold', type=float, default=.25,
                   help='Released Qwen script uses .25; paper Sec. 5.1 reports .35')
    p.add_argument('--faithact-text-threshold', type=float, default=.20,
                   help='Released Qwen script uses .20; paper Sec. 5.1 reports .25')
    p.add_argument('--standardize-faithact-output',action='store_true',
                   help='Append common answer format; this is a declared controlled-comparison variant')
    p.add_argument('--faithact-prompt', choices=['released_code','paper_appendix'],
                   default='released_code',
                   help='Keep the released-code and paper-appendix refinement protocols separate')
    p.add_argument('--cot-cache', type=Path,
                   help='Completed CoT JSONL to reuse for FaithAct (recorded and hashed in manifest)')
    args = p.parse_args(argv)
    if args.tool_device == 'auto':
        args.tool_device_requested = 'auto'
        args.tool_device = os.environ.get('CVPR_TOOL_DEVICE', '')
        if not args.tool_device.startswith('cuda:'):
            p.error('--tool-device auto requires a dedicated tool GPU from queue_bf16')
    methods = args.methods.split(',')
    if not set(methods) <= {'direct', 'cot', 'faithact', 'online', 'online_cf',
                            'online_repair', 'online_cf_repair',
                            'evidence_first', 'evidence_first_cf'}:
        p.error('Unknown method')
    if 'faithact' in methods and not all([args.upstream, args.clip, args.dino, args.poll_checkpoint]):
        p.error('FaithAct requires upstream, clip, dino and poll-checkpoint')
    if 'faithact' in methods:
        # The resident queue predates some protocol variants. Reload explicitly
        # at the job boundary, never during an active case, then transplant the
        # lightweight class logic onto any already-loaded frozen tools below.
        import importlib
        faithact_module = importlib.reload(importlib.import_module('trainingfree.faithact'))
        FaithAct = faithact_module.FaithAct
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest['dataset_sha256'] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if args.cot_cache:
        manifest['cot_cache_sha256'] = hashlib.sha256(args.cot_cache.read_bytes()).hexdigest()
    manifest['metric'] = 'strict_normalized_exact_match; unanswered/errors count incorrect'
    manifest['parser_version'] = PARSER_VERSION
    manifest['protocol'] = 'engineering_port_v1_not_yet_paper_matched'
    manifest['code_sha256'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in sorted(Path(__file__).parent.glob('*.py'))}
    # Never mix changed configurations into a resumed run.
    path = args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Existing run has a different manifest; choose a new output directory')
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    snapshot = args.output/'code_snapshot'
    snapshot.mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob('*.py'):
        target = snapshot/source.name
        if not target.exists():
            shutil.copy2(source, target)
    cases = [json.loads(line) for line in args.dataset.read_text(encoding='utf-8').splitlines() if line.strip()]
    cases = cases[args.offset:args.offset+args.limit if args.limit else None]
    if len({str(c['id']) for c in cases}) != len(cases):
        raise ValueError('Duplicate case IDs')
    cot_cache = {}
    if args.cot_cache:
        cot_cache = {str(row['id']): row for row in
                     map(json.loads, args.cot_cache.read_text(encoding='utf-8').splitlines())}
        missing = [str(case['id']) for case in cases if not cot_cache.get(str(case['id']), {}).get('raw')]
        if missing:
            raise ValueError(f'CoT cache is missing {len(missing)} selected cases, first={missing[0]}')
    from PIL import Image
    from .qwen import Qwen
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(0)
    shared = {} if shared is None else shared
    if 'vlm' not in shared:
        shared['vlm'] = Qwen(args.model, args.device, args.max_pixels, args.max_new_tokens, args.quantization)
    vlm = shared['vlm']
    if vlm.signature != (args.model, args.device, args.max_pixels, args.quantization):
        raise ValueError('Resident worker model configuration does not match requested job')
    vlm.__class__ = Qwen  # Reload stateless adapter methods without reloading frozen weights.
    vlm.max_new_tokens = args.max_new_tokens
    runtime = {'device_map':getattr(vlm.model,'hf_device_map',{}),
               'tool_device':args.tool_device,
               'tool_gpu_budget_mib':os.environ.get('CVPR_TOOL_GPU_BUDGET_MIB'),
               'torch_version':torch.__version__,
               'parameter_dtypes':dict(Counter(str(p.dtype) for p in vlm.model.parameters())),
               'max_pixels':vlm.processor.image_processor.max_pixels,
               'min_pixels':vlm.processor.image_processor.min_pixels}
    (args.output/'runtime_model.json').write_text(json.dumps(runtime,indent=2,default=str))
    summary = {}
    for method in methods:
        if method == 'faithact':
            if 'faithact' not in shared:
                shared['faithact'] = FaithAct(vlm, args.upstream, args.clip, args.dino,
                                              args.poll_checkpoint, args.tool_device,
                                              args.faithact_box_threshold,
                                              args.faithact_text_threshold,
                                              prompt_protocol=args.faithact_prompt)
            faithact = shared['faithact']
            # The resident worker reuses heavy frozen tools across sequential
            # jobs; the protocol remains job-scoped and is captured in manifest.
            faithact.__class__ = FaithAct
            faithact.prompt_protocol = args.faithact_prompt
            faithact.box_threshold = args.faithact_box_threshold
            faithact.text_threshold = args.faithact_text_threshold
        result_path = args.output/(method+'.jsonl')
        prior = [json.loads(s) for s in result_path.read_text(encoding='utf-8').splitlines()] if result_path.exists() else []
        done = {str(r['id']) for r in prior}
        records = list(prior)
        if method in ('evidence_first', 'evidence_first_cf'):
            from .evidence_first import EvidenceFirstOnlineRepair
            backend = EvidenceFirstOnlineRepair(
                vlm, method == 'evidence_first_cf', args.support, args.specificity,
                initial_style=args.initial_style, repair_style=args.repair_style)
        elif method in ('online_repair', 'online_cf_repair'):
            from .online_repair import OnlineRepair
            backend = OnlineRepair(vlm, method == 'online_cf_repair', args.support, args.specificity,
                                   initial_style=args.initial_style, repair_style=args.repair_style)
        else:
            backend = Online(vlm, method == 'online_cf', args.support, args.specificity,
                             initial_style=args.initial_style)
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
                    if method.startswith('online') or method.startswith('evidence_first'):
                        with (args.output/'events.jsonl').open('a',encoding='utf-8') as events:
                            def emit(event):
                                events.write(json.dumps({'id':case['id'],'method':method,
                                                         'time':time.time(),**event},ensure_ascii=False)+'\n')
                                events.flush()
                            record = solve(backend, image, question, args.max_steps, args.repairs, on_event=emit)
                            if getattr(backend, 'last_finalization', None):
                                record['finalization'] = backend.last_finalization
                            if method.startswith('evidence_first'):
                                record['controller_protocol'] = backend.protocol_version
                                record['anchor_raw'] = backend.anchor_raw
                                record['anchor_answer'] = backend.anchor_answer
                    elif method == 'direct':
                        record = {'raw': vlm.generate(image, question+'\nReturn only the concise answer.'), 'status': 'direct'}
                    else:
                        prompt = cot_prompt(question, args.cot_style)
                        if method == 'faithact':
                            cot_path = args.cot_cache or (args.output/'cot.jsonl')
                            cots = cot_cache or ({str(r['id']): r for r in
                                                  map(json.loads, cot_path.read_text(encoding='utf-8').splitlines())}
                                                 if cot_path.exists() else {})
                            cached = cots.get(str(case['id']), {})
                            cot = cached.get('raw') or vlm.generate(image, prompt)
                            if cached.get('raw'):
                                cached_cot_cost = {key: cached.get(key, 0) for key in cached_cot_cost}
                            if args.standardize_faithact_output:
                                from .format_adapter import FinalFormatAdapter
                                faithact.vlm = FinalFormatAdapter(vlm,question)
                            try:
                                record = faithact.run(image, question, cot)
                            finally:
                                faithact.vlm = vlm
                            record['answer_format_protocol'] = 'standardized_final_instruction' if args.standardize_faithact_output else 'released_template'
                            record['initial_cot_reused'] = bool(cached.get('raw'))
                        else:
                            record = {'raw': vlm.generate(image, prompt), 'status': 'cot'}
                    answer = extract_answer(raw_answer(record), question)
                except Exception as exc:
                    # Persist failures; never report accuracy only on successful cases.
                    record, answer = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}',
                                      'traceback': traceback.format_exc()}, ''
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
                           'status_counts':dict(Counter(r['status'] for r in records)),
                           'mean_seconds': sum(r['elapsed_seconds'] for r in records)/max(1,len(records))}
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
