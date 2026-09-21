"""Summarize measured trajectories and expose concrete runtime failure modes."""
import argparse
from collections import Counter
import json
from pathlib import Path


def analyze(folder):
    report = {}
    for path in sorted(folder.glob('*.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        statuses, audits, errors, reasons = Counter(), Counter(), Counter(), Counter()
        examples = []
        for row in rows:
            statuses[row.get('status')] += 1
            for event in row.get('trace', []):
                if 'parse_error' in event:
                    errors[event['parse_error'].split('; raw candidate=')[0]] += 1
                    if len(examples) < 4:
                        examples.append({'id':row['id'], 'error':event['parse_error']})
                if 'audit' in event:
                    audits[event['audit']['status']] += 1
                    reasons[event['audit'].get('reason','')] += 1
                    if len(examples) < 4:
                        examples.append({'id':row['id'], 'candidate':event['candidate'], 'audit':event['audit']})
        report[path.stem] = {'n':len(rows), 'correct':sum(r.get('correct',False) for r in rows),
                            'statuses':dict(statuses), 'audit_statuses':dict(audits),
                            'parse_errors':dict(errors), 'audit_reasons':dict(reasons),
                            'examples':examples,
                            'answers':[{'id':r['id'], 'answer':r.get('answer'), 'reference':r.get('reference'),
                                        'raw_tail':r.get('raw','')[-600:]} for r in rows[:4]]}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.folder), indent=2, ensure_ascii=False))
