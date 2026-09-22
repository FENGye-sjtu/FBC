"""Measured paired accuracy and repair behavior; never alter saved predictions."""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import statistics

from .answers import PARSER_VERSION, answer_source, extract_answer, raw_answer
from .analyze_online import first_attempt, score_text
from .run import normalize


def summarize(folder, dataset):
    cases = {str(c['id']):c for c in map(json.loads,dataset.read_text(encoding='utf-8').splitlines())}
    manifest = json.loads((folder/'manifest.json').read_text())
    offset, limit = manifest.get('offset',0), manifest.get('limit',0)
    expected_n = min(limit or len(cases),max(0,len(cases)-offset))
    report = {'run':str(folder),'manifest':manifest,
              'scorer_sha256':hashlib.sha256(Path(inspect.getsourcefile(extract_answer)).read_bytes()).hexdigest(),
              'parser_version':PARSER_VERSION,
              'note':'Development engineering results unless full protocol reconciliation is documented.',
              'methods':{},'paired':{}}
    indexed = {}
    for method in manifest['methods'].split(','):
        path = folder/(method+'.jsonl')
        rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
        audited, repaired, cf, changes = 0, 0, 0, Counter()
        corrected, scores, repair_attempts, uncertainty, initial_missing = [], {}, [], 0, 0
        for row in rows:
            case = cases[str(row['id'])]
            # Use only prediction and question to extract. Reference is compared afterwards.
            answer = extract_answer(raw_answer(row),case['question'])
            correct = bool(answer) and normalize(answer)==normalize(case['reference'])
            scores[str(row['id'])] = correct
            if isinstance(row.get('correct'),bool) and correct != row['correct']:
                corrected.append({'id':row['id'],'old_answer':row.get('answer'),'extracted_answer':answer,
                                  'old_correct':row['correct'],'correct':correct,
                                  'answer_source':answer_source(row)})
            trace = row.get('trace',[])
            first = first_attempt(trace).get('candidate') or {}
            first_answer = first.get('final_answer')
            _, first_ok = score_text(first_answer,case['question'],case['reference'])
            if first_ok is not None:
                changes[f'{"correct" if first_ok else "wrong"}_to_{"correct" if correct else "wrong"}'] += 1
            else:
                initial_missing += 1
            for event in trace:
                if 'audit' in event:
                    audited += 1
                    audit = event['audit']
                    cf += 'target_mask' in audit['scores']
                    repaired += event['attempt']>0 and audit['status']=='supported'
                    repair_attempts.append(event['attempt'])
                uncertainty += event.get('event')=='uncertainty_commit'
        n = len(rows)
        report['methods'][method] = {
            'n':n,'expected_n':expected_n,'complete':n==expected_n,
            'correct':sum(scores.values()),'accuracy':sum(scores.values())/n if n else None,
            'stored_correct_before_parser_update':sum(r.get('correct') is True for r in rows),
            'stored_scored_n':sum(isinstance(r.get('correct'),bool) for r in rows),
            'parser_changes':corrected,'status_counts':dict(Counter(r['status'] for r in rows)),
            'audits':audited,'counterfactual_audits_executed':cf,'accepted_after_repair':repaired,
            'uncertain_points_committed':uncertainty,'max_repair_index':max(repair_attempts,default=0),
            'initial_proposed_answer_to_final':dict(changes),
            'initial_valid_answer_n':n-initial_missing,'initial_missing_answer_n':initial_missing,
            'mean_seconds':statistics.mean(r['elapsed_seconds'] for r in rows) if rows else None,
            'mean_model_calls':statistics.mean(r['calls'] for r in rows) if rows else None}
        indexed[method] = scores
    for left,right in [('cot','faithact'),('faithact','online'),('online','online_cf'),('faithact','online_cf')]:
        common = indexed.get(left,{}).keys() & indexed.get(right,{}).keys()
        if common:
            wins = sorted((i for i in common if indexed[right][i] and not indexed[left][i]),key=int)
            losses = sorted((i for i in common if indexed[left][i] and not indexed[right][i]),key=int)
            report['paired'][f'{right}_vs_{left}'] = {'n':len(common),'wins':wins,'losses':losses,
                                                     'net_correct':len(wins)-len(losses)}
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('folder',type=Path)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    report = summarize(args.folder,args.dataset)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'parser_version':report['parser_version'],
                      'methods':report['methods'],'paired':report['paired']},indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
