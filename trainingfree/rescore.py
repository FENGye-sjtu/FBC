"""Re-evaluate saved raw answers with a reference-blind parser, without rewriting traces."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from .answers import PARSER_VERSION, answer_source, extract_answer, raw_answer
from .run import normalize


def rescore(run, dataset):
    run, dataset = Path(run), Path(dataset)
    cases = {str(r['id']):r for r in map(json.loads,dataset.read_text(encoding='utf-8').splitlines())}
    report = {'source_run':str(run), 'parser_sha256':hashlib.sha256(Path(inspect.getsourcefile(extract_answer)).read_bytes()).hexdigest(),
              'parser_version':PARSER_VERSION,
              'note':'Engineering pilot; parser correction only, original records preserved.', 'methods':{}}
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    for method in manifest['methods'].split(','):
        path = run/(method+'.jsonl')
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        details = []
        for row in rows:
            case = cases[str(row['id'])]
            answer = extract_answer(raw_answer(row),case['question'])
            details.append({'id':row['id'],'previous_answer':row.get('answer'),'answer':answer,
                            'answer_source':answer_source(row),
                            'correct':bool(answer) and normalize(answer)==normalize(case['reference'])})
        report['methods'][path.stem] = {'n':len(rows),
                                       'previous_correct':sum(r.get('correct') is True for r in rows),
                                       'previous_scored_n':sum(isinstance(r.get('correct'),bool) for r in rows),
                                       'correct':sum(r['correct'] for r in details),'cases':details}
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = rescore(args.run,args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
