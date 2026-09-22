"""Reproducible adapters for third-party RealWorldQA evaluation protocols.

The adapters never copy or silently modify upstream scoring rules.  VLMEvalKit's
``can_infer`` is loaded directly from a pinned checkout so that its exact-match
policy can run without importing every optional dataset dependency.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import string
import sys
import types


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def read_tsv(path):
    csv.field_size_limit(sys.maxsize)
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def choices_from(item):
    return {letter: item[letter] for letter in string.ascii_uppercase
            if item.get(letter, '').strip()}


def load_vlmeval_can_infer(toolkit_root):
    """Load the unmodified pinned VLMEvalKit matching function.

    Importing ``vlmeval`` currently imports all optional benchmark modules.  The
    matching file itself only needs a logger, so provide exactly that interface
    and execute the upstream source file verbatim.
    """
    source = Path(toolkit_root)/'vlmeval/utils/matching_util.py'
    if not source.is_file():
        raise FileNotFoundError(source)

    class Logger:
        def info(self, *args, **kwargs):
            pass

    saved = {name: sys.modules.get(name) for name in
             ('vlmeval', 'vlmeval.smp', 'vlmeval.smp.log')}
    vlmeval = types.ModuleType('vlmeval')
    smp = types.ModuleType('vlmeval.smp')
    log = types.ModuleType('vlmeval.smp.log')
    log.get_logger = lambda name: Logger()
    sys.modules.update({'vlmeval': vlmeval, 'vlmeval.smp': smp,
                        'vlmeval.smp.log': log})
    try:
        spec = importlib.util.spec_from_file_location('_pinned_vlmeval_matching', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module.can_infer, source


def vlmeval_score(raw_path, tsv_path, mapping_path, toolkit_root, output_dir):
    raw = {str(row['id']): row for row in jsonl(raw_path)}
    meta = read_tsv(tsv_path)
    mapping = (json.loads(Path(mapping_path).read_text(encoding='utf-8')) if mapping_path else
               [{'vlm_index': str(row['index']), 'xai_id': str(row['index'])} for row in meta])
    by_index = {str(row['index']): row for row in meta}
    can_infer, source = load_vlmeval_can_infer(toolkit_root)
    records = []
    for link in mapping:
        index, case_id = str(link['vlm_index']), str(link['xai_id'])
        item, prediction = by_index[index], raw[case_id].get('raw', '')
        choices = choices_from(item)
        inferred = can_infer(prediction, choices.copy()) or 'Z'
        reference = str(item['answer'])
        records.append({'index': index, 'xai_id': case_id, 'prediction': prediction,
                        'inferred': inferred, 'reference': reference,
                        'correct': inferred == reference})
    if len(records) != len(meta) or len({r['index'] for r in records}) != len(records):
        raise ValueError('Mapping must cover each VLMEvalKit row exactly once')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details = output_dir/'details.jsonl'
    details.write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in records),
                       encoding='utf-8')
    correct = sum(x['correct'] for x in records)
    summary = {
        'protocol': 'VLMEvalKit can_infer exact_matching (no LLM judge)',
        'n': len(records), 'correct': correct, 'accuracy': correct/len(records),
        'upstream_matching_sha256': sha256(source),
        'dataset_md5': hashlib.md5(Path(tsv_path).read_bytes()).hexdigest(),
        'raw_sha256': sha256(raw_path),
        'mapping_sha256': sha256(mapping_path) if mapping_path else None,
        'details': str(details),
    }
    (output_dir/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def materialize_vlmeval_cases(tsv_path, mapping_path, original_cases, output_path):
    """Create inference cases with VLMEvalKit prompt packaging and mapped images."""
    meta = read_tsv(tsv_path)
    by_index = {str(row['index']): row for row in meta}
    originals = {str(row['id']): row for row in jsonl(original_cases)}
    mapping = json.loads(Path(mapping_path).read_text(encoding='utf-8'))
    cases = []
    for link in mapping:
        item = by_index[str(link['vlm_index'])]
        original = originals[str(link['xai_id'])]
        choices = list(choices_from(item).items())
        prompt = f"Question: {item['question']}\nOptions:\n"
        prompt += ''.join(f'{letter}. {value}\n' for letter, value in choices)
        prompt += 'Please select the correct answer from the options above.  \n'
        cases.append({'id': str(item['index']), 'image': original['image'],
                      'question': prompt, 'reference': str(item['answer']),
                      'source_xai_id': str(link['xai_id'])})
    if len(cases) != len(meta):
        raise ValueError('Mapping does not cover the full dataset')
    Path(output_path).write_text(
        ''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in cases), encoding='utf-8')
    return {'n': len(cases), 'sha256': sha256(output_path)}


def materialize_ags(raw_path, cases_path, output_path):
    """Convert our trace records to AGS getResult_nlp.py's public JSONL schema."""
    raw = {str(row['id']): row for row in jsonl(raw_path)}
    cases = {str(row['id']): row for row in jsonl(cases_path)}
    if set(raw) != set(cases):
        raise ValueError('AGS conversion requires one prediction for every case')
    rows = []
    for case_id, case in cases.items():
        prediction = raw[case_id].get('raw', raw[case_id].get('answer', ''))
        rows.append({'id': case_id, 'question': case['question'],
                     'answer': str(case.get('reference', case.get('answer', ''))),
                     'model_answer': prediction,
                     'len_ans': raw[case_id].get('generated_tokens', len(prediction.split()))})
    Path(output_path).write_text(
        ''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in rows), encoding='utf-8')
    return {'n': len(rows), 'sha256': sha256(output_path)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    score = sub.add_parser('vlmeval-score')
    for name in ('raw', 'tsv', 'toolkit-root', 'output'):
        score.add_argument('--'+name, required=True, type=Path)
    score.add_argument('--mapping', type=Path,
                       help='Optional VLMEval index to source prediction ID map')
    prepare = sub.add_parser('prepare-vlmeval')
    for name in ('tsv', 'mapping', 'original-cases', 'output'):
        prepare.add_argument('--'+name, required=True, type=Path)
    ags = sub.add_parser('prepare-ags')
    for name in ('raw', 'cases', 'output'):
        ags.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == 'vlmeval-score':
        result = vlmeval_score(args.raw, args.tsv, args.mapping,
                               args.toolkit_root, args.output)
    elif args.command == 'prepare-vlmeval':
        result = materialize_vlmeval_cases(args.tsv, args.mapping,
                                           args.original_cases, args.output)
    else:
        result = materialize_ags(args.raw, args.cases, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
