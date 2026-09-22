"""Reference-blind comparison of prediction files from separate run folders."""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import math

from .answers import PARSER_VERSION, answer_source, extract_answer, raw_answer
from .run import normalize


INPUT_BUDGET_FIELDS = ('model', 'quantization', 'max_pixels', 'max_new_tokens', 'dataset_sha256')


def wilson_interval(correct, n, z=1.96):
    if not n:
        return [None, None]
    p = correct/n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return [center-radius, center+radius]


def mcnemar_exact(wins, losses):
    """Two-sided exact binomial p-value on discordant paired outcomes."""
    n = wins + losses
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses)+1)) / 2**n
    return min(1.0, 2*tail)


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def score_rows(rows, cases):
    decisions, parser_changes = {}, []
    for row in rows:
        case_id = str(row['id'])
        case = cases[case_id]
        answer = extract_answer(raw_answer(row), case['question'])
        correct = bool(answer) and normalize(answer) == normalize(case['reference'])
        decisions[case_id] = correct
        if bool(row.get('correct')) != correct:
            parser_changes.append({'id':case_id, 'stored_answer':row.get('answer',''),
                                   'parsed_answer':answer, 'stored_correct':bool(row.get('correct')),
                                   'correct':correct, 'answer_source':answer_source(row)})
    return decisions, parser_changes


def paired_outcomes(left, right, case_ids):
    """Measure both methods on exactly the same set of cases."""
    common = set(case_ids)
    order = lambda value: (0, int(value)) if value.isdigit() else (1, value)
    wins = sorted((i for i in common if right[i] and not left[i]), key=order)
    losses = sorted((i for i in common if left[i] and not right[i]), key=order)
    return {
        'n': len(common), 'left_correct': sum(left[i] for i in common),
        'right_correct': sum(right[i] for i in common),
        'wins': wins, 'losses': losses, 'net_correct': len(wins)-len(losses),
        'mcnemar_exact_p': mcnemar_exact(len(wins), len(losses))}


def input_budget_differences(left, right, dataset_sha256):
    """Missing run metadata cannot establish a matched experimental budget."""
    differences = {}
    for field in INPUT_BUDGET_FIELDS:
        lvalue, rvalue = left.get(field), right.get(field)
        reason = None
        if lvalue is None or rvalue is None:
            reason = 'missing_manifest_value'
        elif lvalue != rvalue:
            reason = 'different_run_values'
        elif field == 'dataset_sha256' and lvalue != dataset_sha256:
            reason = 'different_evaluation_dataset'
        if reason:
            differences[field] = {'left': lvalue, 'right': rvalue, 'reason': reason}
    return differences


def compare(dataset, predictions):
    cases = {str(row['id']):row for row in load_jsonl(dataset)}
    report = {
        'dataset': str(dataset),
        'dataset_sha256': hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        'parser_sha256': hashlib.sha256(Path(inspect.getsourcefile(extract_answer)).read_bytes()).hexdigest(),
        'parser_version': PARSER_VERSION,
        'reference_blind_extraction': True,
        'methods': {}, 'paired': {}}
    indexed, records, manifests = {}, {}, {}
    for name, path in predictions:
        rows = load_jsonl(path)
        decisions, changes = score_rows(rows, cases)
        indexed[name] = decisions
        records[name] = {str(row['id']): row for row in rows}
        manifest_path = Path(path).parent/'manifest.json'
        manifests[name] = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
        correct = sum(decisions.values())
        report['methods'][name] = {
            'path':str(path), 'n':len(rows), 'correct':sum(decisions.values()),
            'manifest_path': str(manifest_path) if manifest_path.exists() else None,
            'input_budget': {field: manifests[name].get(field) for field in INPUT_BUDGET_FIELDS},
            'accuracy':sum(decisions.values())/len(rows) if rows else None,
            'accuracy_wilson95':wilson_interval(correct, len(rows)),
            'uncertain':sum(row.get('status') == 'answered_with_uncertainty' for row in rows),
            'with_evidence':sum(bool(row.get('evidence')) for row in rows),
            'mean_seconds':statistics.mean(row.get('elapsed_seconds',0) for row in rows) if rows else None,
            'status_counts':dict(Counter(row.get('status','') for row in rows)),
            'parser_changes':changes}
    names = [name for name, _ in predictions]
    for left in names:
        for right in names:
            if names.index(right) <= names.index(left):
                continue
            common = indexed[left].keys() & indexed[right].keys()
            pair = paired_outcomes(indexed[left], indexed[right], common)
            differences = input_budget_differences(manifests[left], manifests[right], report['dataset_sha256'])
            pair['input_budget_matched'] = not differences
            pair['input_budget_differences'] = differences
            pair['comparison_scope'] = 'matched_input_budget' if not differences else 'diagnostic_only_unmatched_input_budget'
            subgroups = {}
            for name in (left, right):
                if not any(row.get('method') == 'faithact' or
                           str(row.get('status', '')).startswith('faithact') or
                           ('original_cot' in row and 'evidence' in row)
                           for row in records[name].values()):
                    continue
                evidence_ids = {i for i in common if records[name][i].get('evidence')}
                subgroups[name] = {
                    'with_evidence': paired_outcomes(indexed[left], indexed[right], evidence_ids),
                    'without_evidence': paired_outcomes(indexed[left], indexed[right], common-evidence_ids)}
            if subgroups:
                pair['evidence_subgroups'] = subgroups
            report['paired'][f'{right}_vs_{left}'] = pair
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--prediction', action='append', required=True,
                        help='NAME=/absolute/path/to/predictions.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    predictions = []
    for item in args.prediction:
        name, separator, path = item.partition('=')
        if not separator or not name or not path:
            parser.error('--prediction must be NAME=PATH')
        predictions.append((name, Path(path)))
    result = compare(args.dataset, predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'parser_version':result['parser_version'],
                     'methods':result['methods'], 'paired':result['paired']},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
