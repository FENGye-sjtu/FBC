"""Reference-blind comparison of prediction files from separate run folders."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import math

from .run import extract_answer, normalize


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
        answer = extract_answer(row.get('raw', row.get('answer', '')), case['question'])
        correct = bool(answer) and normalize(answer) == normalize(case['reference'])
        decisions[case_id] = correct
        if bool(row.get('correct')) != correct:
            parser_changes.append({'id':case_id, 'stored_answer':row.get('answer',''),
                                   'parsed_answer':answer, 'stored_correct':bool(row.get('correct')),
                                   'correct':correct})
    return decisions, parser_changes


def compare(dataset, predictions):
    cases = {str(row['id']):row for row in load_jsonl(dataset)}
    report = {
        'dataset': str(dataset),
        'dataset_sha256': hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        'parser_sha256': hashlib.sha256(Path(__file__).with_name('run.py').read_bytes()).hexdigest(),
        'reference_blind_extraction': True,
        'methods': {}, 'paired': {}}
    indexed = {}
    for name, path in predictions:
        rows = load_jsonl(path)
        decisions, changes = score_rows(rows, cases)
        indexed[name] = decisions
        correct = sum(decisions.values())
        report['methods'][name] = {
            'path':str(path), 'n':len(rows), 'correct':sum(decisions.values()),
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
            wins = sorted((i for i in common if indexed[right][i] and not indexed[left][i]), key=int)
            losses = sorted((i for i in common if indexed[left][i] and not indexed[right][i]), key=int)
            report['paired'][f'{right}_vs_{left}'] = {
                'n':len(common), 'wins':wins, 'losses':losses,
                'net_correct':len(wins)-len(losses),
                'mcnemar_exact_p':mcnemar_exact(len(wins), len(losses))}
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
    print(json.dumps({'methods':result['methods'], 'paired':result['paired']},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
