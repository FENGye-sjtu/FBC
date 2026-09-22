"""Re-score fixed raw outputs with two reference-blind answer extractors."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

from .answers import PARSER_VERSION, answer_source, extract_answer, raw_answer


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(paths, dataset, legacy_extract=None):
    """Return all disagreements, gains AND losses without modifying raw files."""
    if legacy_extract is None:
        from .run import legacy_extract_answer
        legacy_extract = legacy_extract_answer
    legacy_path = inspect.getsourcefile(legacy_extract)
    legacy_path = Path(legacy_path) if legacy_path and Path(legacy_path).is_file() else None
    dataset = Path(dataset)
    cases = {str(case['id']): case for case in
             map(json.loads, dataset.read_text(encoding='utf-8').splitlines())}
    report = {'parser_version': PARSER_VERSION,
              'dataset': str(dataset), 'dataset_sha256': _sha(dataset),
              'parser_sha256': _sha(Path(__file__).with_name('answers.py')),
              'legacy_parser_version': (legacy_extract.__globals__.get('PARSER_VERSION',
                                                                       legacy_extract.__name__)
                                        if legacy_extract.__name__ == 'extract_answer'
                                        else legacy_extract.__name__),
              'legacy_source': str(legacy_path) if legacy_path else None,
              'legacy_source_sha256': _sha(legacy_path) if legacy_path else None,
              'files': []}
    for path in paths:
        path = Path(path)
        snapshot = path.read_bytes()
        rows = [json.loads(line) for line in snapshot.decode('utf-8').splitlines()
                if line.strip()]
        old_correct = new_correct = 0
        old_unresolved = new_unresolved = 0
        disagreements, recovered, lost = [], [], []
        for row in rows:
            case = cases[str(row['id'])]
            raw = raw_answer(row)
            question = case['question']
            reference = str(case.get('reference', case.get('answer', ''))).strip()
            # References are used only after both independent extractions.
            old, new = legacy_extract(raw, question), extract_answer(raw, question)
            old_ok = bool(old) and str(old).strip().casefold() == reference.casefold()
            new_ok = bool(new) and str(new).strip().casefold() == reference.casefold()
            old_correct += old_ok
            new_correct += new_ok
            old_unresolved += old not in {'A', 'B', 'C', 'D', 'E'}
            new_unresolved += new not in {'A', 'B', 'C', 'D', 'E'}
            if old != new:
                disagreements.append({'id': row['id'], 'old': old, 'new': new,
                                      'reference': reference, 'old_correct': old_ok,
                                      'new_correct': new_ok, 'raw': raw,
                                      'answer_source': answer_source(row)})
            if new_ok and not old_ok:
                recovered.append(row['id'])
            if old_ok and not new_ok:
                lost.append(row['id'])
        count = len(rows)
        report['files'].append({'path': str(path), 'sha256': hashlib.sha256(snapshot).hexdigest(), 'n': count,
                                'old_correct': old_correct, 'new_correct': new_correct,
                                'old_accuracy': old_correct / count if count else None,
                                'new_accuracy': new_correct / count if count else None,
                                'old_unresolved': old_unresolved, 'new_unresolved': new_unresolved,
                                'recovered_ids': recovered, 'lost_ids': lost,
                                'disagreements': disagreements})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--legacy-source', type=Path,
                        help='Immutable parser module to compare, e.g. the published v1 answers.py')
    args = parser.parse_args(argv)
    legacy_extract = None
    if args.legacy_source:
        spec = importlib.util.spec_from_file_location('_parser_audit_legacy', args.legacy_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        legacy_extract = module.extract_answer
    report = audit(args.input, args.dataset, legacy_extract=legacy_extract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'files': [
        {key: value for key, value in result.items() if key != 'disagreements'}
        for result in report['files']]}, indent=2))


if __name__ == '__main__':
    main()
