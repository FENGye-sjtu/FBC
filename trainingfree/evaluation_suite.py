"""Audit complete or explicitly partial prediction files without changing raw outputs."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys

from .answers import PARSER_VERSION, answer_source, choices_from_question, extract_answer, raw_answer
from .eval_protocols import load_vlmeval_can_infer
from .run import normalize


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_rows(path, kind, snapshot=None):
    indexed = {}
    text = (Path(path).read_bytes() if snapshot is None else snapshot).decode('utf-8')
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if 'id' not in row:
            raise ValueError(f'{kind} line {line_number} has no id')
        case_id = str(row['id'])
        if case_id in indexed:
            raise ValueError(f'Duplicate {kind} ID: {case_id}')
        indexed[case_id] = row
    return indexed


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows),
                          encoding='utf-8')


def run_ags(ags_root, records, cases, output):
    """Run the released evaluator verbatim and retain its actual stdout counts."""
    source = Path(ags_root).resolve()/'extract_ans/getResult_nlp.py'
    if not source.is_file():
        raise FileNotFoundError(source)
    materialized = output/'ags_input.jsonl'
    write_jsonl(materialized, [
        {'id': case_id, 'question': cases[case_id]['question'],
         'answer': str(cases[case_id]['reference']), 'model_answer': raw_answer(row),
         'len_ans': row.get('generated_tokens', len(raw_answer(row).split()))}
        for case_id, row in records.items()])
    command = [sys.executable, str(source), '--input-files', str(materialized.resolve()),
               '--show-errors', str(len(records))]
    completed = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                               errors='replace', cwd=str(output.resolve()), check=False)
    stdout_path, stderr_path = output/'ags_stdout.txt', output/'ags_stderr.txt'
    stdout_path.write_text(completed.stdout, encoding='utf-8')
    stderr_path.write_text(completed.stderr, encoding='utf-8')
    if completed.returncode:
        raise RuntimeError(f'AGS exited {completed.returncode}; inspect {stderr_path}')
    totals = re.findall(r'^\s*Total:\s*(\d+)\s*$', completed.stdout, re.M)
    correct = re.findall(r'^\s*Correct:\s*(\d+)\s*$', completed.stdout, re.M)
    if len(totals) != 1 or len(correct) != 1:
        raise ValueError(f'Expected one AGS Total/Correct result; inspect {stdout_path}')
    total, hits = int(totals[0]), int(correct[0])
    if total != len(records) or not 0 <= hits <= total:
        raise ValueError(f'AGS coverage/count mismatch: Total={total}, Correct={hits}, input={len(records)}')
    return {
        'protocol': 'unmodified AGS extract_ans/getResult_nlp.py on raw model output',
        'n': total, 'correct': hits, 'accuracy': hits/total if total else None,
        'per_case_available': False,
        'per_case_note': 'Released stdout is preserved; per-case decisions are not reimplemented.',
        'source': str(source), 'source_sha256': sha256(source),
        'input': str(materialized), 'input_sha256': sha256(materialized),
        'stdout': str(stdout_path), 'stdout_sha256': sha256(stdout_path),
        'stderr': str(stderr_path), 'stderr_sha256': sha256(stderr_path),
        'command': command}


def evaluate(predictions, cases_path, output, allow_partial=False, ags_root=None, vlmeval_root=None):
    """Score a validated ID set with one frozen reference-blind parser."""
    predictions, cases_path, output = Path(predictions), Path(cases_path), Path(output)
    # A live generation job may append after this read. Evaluate and hash these
    # exact bytes, rather than hashing a later and possibly longer file.
    raw_snapshot, dataset_snapshot = predictions.read_bytes(), cases_path.read_bytes()
    cases = unique_rows(cases_path, 'dataset', dataset_snapshot)
    records = unique_rows(predictions, 'prediction', raw_snapshot)
    if not cases:
        raise ValueError('Dataset is empty')
    if not records:
        raise ValueError('Prediction file is empty')
    extra = sorted(records.keys()-cases.keys())
    missing = sorted(cases.keys()-records.keys())
    if extra:
        raise ValueError(f'Unknown prediction IDs: {extra[:10]}')
    if missing and not allow_partial:
        raise ValueError(f'Incomplete prediction coverage: {len(records)}/{len(cases)}; '
                         'use --allow-partial only for an explicitly partial report')
    for case_id, case in cases.items():
        if 'question' not in case or 'reference' not in case:
            raise ValueError(f'Dataset ID {case_id} needs question and reference')
    generated = ('raw_predictions.jsonl', 'cases.jsonl', 'summary.json', 'details.jsonl',
                 'disagreements.jsonl', 'ags_input.jsonl', 'ags_stdout.txt', 'ags_stderr.txt')
    protected = {predictions.resolve(), cases_path.resolve()}
    if any((output/name).resolve() in protected for name in generated):
        raise ValueError('Evaluation output would overwrite a source prediction or dataset file')
    output.mkdir(parents=True, exist_ok=True)
    (output/'raw_predictions.jsonl').write_bytes(raw_snapshot)
    (output/'cases.jsonl').write_bytes(dataset_snapshot)
    can_infer, matching_source = (load_vlmeval_can_infer(vlmeval_root) if vlmeval_root else (None, None))
    details = []
    for case_id, row in records.items():
        case = cases[case_id]
        text, reference = raw_answer(row), str(case['reference'])
        choices = choices_from_question(case['question'])
        extracted = extract_answer(text, case['question'])
        answer = '' if extracted is None else str(extracted)
        is_correct = bool(answer) and normalize(answer) == normalize(reference)
        source_cot = row.get('original_cot', text)
        source_cot = '' if source_cot is None else str(source_cot)
        interior_blocks = [block for block in source_cot.split('\n\n')[1:-1] if block.strip()]
        scores = {'committed_answer': {'answer': answer, 'correct': is_correct,
                                      'valid_label': answer in choices if choices else None}}
        if can_infer is not None:
            if not choices:
                raise ValueError(f'VLMEval MCQ scoring requires public options, missing for ID {case_id}')
            if reference not in choices:
                raise ValueError(f'VLMEval reference must be an exact public option label for ID {case_id}')
            for name, value in (('vlmeval_raw', text), ('vlmeval_canonical_final', answer)):
                upstream_answer = can_infer(value, choices.copy())
                inferred = upstream_answer or 'Z'
                scores[name] = {'answer': inferred, 'correct': inferred == reference,
                                'valid_label': inferred in choices, 'upstream_answer': upstream_answer}
        stored_change = None
        if 'correct' in row:
            stored_change = bool(row['correct']) != is_correct
        stored_answer_change = (normalize(row['answer']) != normalize(answer)) if 'answer' in row else None
        detail = {
            'id': case_id, 'question': case['question'], 'reference': reference,
            'raw': text, 'stored_answer': row.get('answer'), 'stored_correct': row.get('correct'),
            'answer_source': answer_source(row),
            'status': row.get('status'), 'with_evidence': bool(row.get('evidence')),
            'faithact_interior_cot_blocks': len(interior_blocks),
            'cot_source': 'original_cot' if 'original_cot' in row else answer_source(row),
            'scorers': scores, 'stored_correctness_changed': stored_change,
            'stored_answer_changed': stored_answer_change,
            'answer_disagreement': len({str(score['answer']) for score in scores.values()}) > 1,
            'correctness_disagreement': len({score['correct'] for score in scores.values()}) > 1}
        details.append(detail)
    details_path, disagreements_path = output/'details.jsonl', output/'disagreements.jsonl'
    disagreements = [row for row in details if row['answer_disagreement'] or row['stored_answer_changed']
                     or row['stored_correctness_changed']]
    write_jsonl(details_path, details)
    write_jsonl(disagreements_path, disagreements)
    source_functions = {'extractor': extract_answer, 'normalizer': normalize,
                        'vlmeval_loader': load_vlmeval_can_infer, 'evaluation_suite': evaluate}
    sources = {name: {'path': inspect.getsourcefile(function),
                      'sha256': sha256(inspect.getsourcefile(function))}
               for name, function in source_functions.items()}
    scorers = {}
    for name in details[0]['scorers']:
        hits = sum(row['scorers'][name]['correct'] for row in details)
        scorers[name] = {
            'n': len(details), 'correct': hits, 'accuracy': hits/len(details),
            'valid_label_n': sum(row['scorers'][name]['valid_label'] is True for row in details)}
    if matching_source:
        sources['vlmeval_matching'] = {'path': str(matching_source), 'sha256': sha256(matching_source)}
    if ags_root:
        scorers['ags_raw'] = run_ags(ags_root, records, cases, output)
        sources['ags_evaluator'] = {'path': scorers['ags_raw']['source'],
                                    'sha256': scorers['ags_raw']['source_sha256']}
    summary = {
        'expected_n': len(cases), 'completed_n': len(records), 'is_partial': bool(missing),
        'allow_partial': allow_partial, 'missing_ids': missing,
        'correct': scorers['committed_answer']['correct'],
        'accuracy': scorers['committed_answer']['accuracy'],
        'valid_label_n': scorers['committed_answer']['valid_label_n'],
        'mcq_n': sum(bool(choices_from_question(cases[case_id]['question'])) for case_id in records),
        'with_evidence_n': sum(row['with_evidence'] for row in details),
        'with_interior_cot_n': sum(row['faithact_interior_cot_blocks'] > 0 for row in details),
        'interior_cot_block_n': sum(row['faithact_interior_cot_blocks'] for row in details),
        'stored_correctness_changed_n': sum(row['stored_correctness_changed'] is True for row in details),
        'stored_answer_changed_n': sum(row['stored_answer_changed'] is True for row in details),
        'answer_disagreement_n': sum(row['answer_disagreement'] for row in details),
        'correctness_disagreement_n': sum(row['correctness_disagreement'] for row in details),
        'disagreement_n': len(disagreements), 'reference_blind_extraction': True,
        'parser_version': PARSER_VERSION,
        'canonical_final_protocol': 'VLMEval can_infer receives only reference-blind extracted answer',
        'predictions': str(predictions), 'raw_sha256': hashlib.sha256(raw_snapshot).hexdigest(),
        'raw_snapshot': str(output/'raw_predictions.jsonl'),
        'dataset': str(cases_path), 'dataset_sha256': hashlib.sha256(dataset_snapshot).hexdigest(),
        'dataset_snapshot': str(output/'cases.jsonl'),
        'details': str(details_path), 'details_sha256': sha256(details_path),
        'disagreements': str(disagreements_path), 'disagreements_sha256': sha256(disagreements_path),
        'sources': sources, 'scorers': scorers}
    manifest = predictions.parent/'manifest.json'
    if manifest.is_file():
        summary['run_manifest'] = str(manifest)
        summary['run_manifest_sha256'] = sha256(manifest)
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', '--raw', required=True, type=Path)
    parser.add_argument('--cases', '--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--ags-root', type=Path)
    parser.add_argument('--vlmeval-root', type=Path)
    args = parser.parse_args(argv)
    result = evaluate(args.predictions, args.cases, args.output, args.allow_partial,
                      args.ags_root, args.vlmeval_root)
    print(json.dumps({
        'completed_n': result['completed_n'], 'expected_n': result['expected_n'],
        'is_partial': result['is_partial'], 'correct': result['correct'],
        'accuracy': result['accuracy'], 'valid_label_n': result['valid_label_n'],
        'with_evidence_n': result['with_evidence_n'],
        'with_interior_cot_n': result['with_interior_cot_n'],
        'disagreement_n': result['disagreement_n'],
        'scorers': {name: {key: score[key] for key in ('n','correct','accuracy')}
                    for name, score in result['scorers'].items()},
        'summary': str(args.output/'summary.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
