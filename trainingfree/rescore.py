"""Re-evaluate saved raw answers with a reference-blind parser, without rewriting traces."""
import argparse
import hashlib
import json
from pathlib import Path
from .run import extract_answer, normalize


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    questions = {str(r['id']):r['question'] for r in map(json.loads,args.dataset.read_text().splitlines())}
    report = {'source_run':str(args.run), 'parser_sha256':hashlib.sha256(Path(__file__).with_name('run.py').read_bytes()).hexdigest(),
              'note':'Engineering pilot; parser correction only, original records preserved.', 'methods':{}}
    manifest = json.loads((args.run/'manifest.json').read_text())
    for method in manifest['methods'].split(','):
        path = args.run/(method+'.jsonl')
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        details = []
        for row in rows:
            answer = extract_answer(row.get('raw',row.get('answer')), questions[str(row['id'])])
            details.append({'id':row['id'],'previous_answer':row['answer'],'answer':answer,
                            'correct':bool(answer) and normalize(answer)==normalize(row['reference'])})
        report['methods'][path.stem] = {'n':len(rows),'previous_correct':sum(r['correct'] for r in rows),
                                       'correct':sum(r['correct'] for r in details),'cases':details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
