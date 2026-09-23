"""Frozen, reference-blind answer-consistency gate for diagnostic comparison.

This is not by itself an Evidence-First method result: it has no step ledger.
Use it to assess whether option likelihood can serve as a useful deduction gate.
"""

import argparse
import json
from pathlib import Path


MARGIN = 0.75  # Fixed on NF4 development IDs 0–127 before BF16 validation.


def load_rows(path):
    return {str(row['id']): row for row in
            (json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
             if line.strip())}


def select(base, option, margin=MARGIN):
    scores = sorted((float(x) for x in option['scores'].values()), reverse=True)
    gap = scores[0] - scores[1]
    change = option['answer'] != base['answer'] and gap >= margin
    return {'id': str(base['id']), 'answer': option['answer'] if change else base['answer'],
            'base_answer': base['answer'], 'option_answer': option['answer'],
            'option_margin': gap, 'changed': change,
            'reference': base.get('reference'),
            'correct': (option['answer'] if change else base['answer']) == base.get('reference'),
            'method': 'option_margin_diagnostic', 'threshold': margin}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--option-scores', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = load_rows(args.base)
    options = load_rows(args.option_scores)
    if base.keys() != options.keys():
        raise ValueError('Paired files must have exactly the same case IDs')
    rows = [select(base[case_id], options[case_id]) for case_id in base]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
                           encoding='utf-8')
    print(json.dumps({'n': len(rows), 'correct': sum(row['correct'] for row in rows),
                      'changed': sum(row['changed'] for row in rows), 'threshold': MARGIN}))


if __name__ == '__main__':
    main()
