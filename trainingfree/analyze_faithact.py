"""Diagnose fixed FaithAct outputs against their exact initial CoT baseline."""
import argparse
from collections import Counter
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics

from .answers import PARSER_VERSION, choices_from_question, extract_answer, raw_answer
from .compare import paired_outcomes
from .run import normalize


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_unique(path, role):
    records = {}
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or row.get('id') is None:
            raise ValueError(f'{role} line {number} has no valid ID')
        case_id = str(row['id'])
        if case_id in records:
            raise ValueError(f'Duplicate ID {case_id!r} in {role}')
        records[case_id] = row
    return records


def _ordered(ids):
    return sorted(ids, key=lambda value: (0, int(value)) if value.isdigit() else (1, value))


def _timing(records, ids, field):
    values = [records[i].get(field) for i in ids]
    values = [value for value in values if isinstance(value, (float, int))
              and not isinstance(value, bool) and math.isfinite(value) and value >= 0]
    return {'recorded_n': len(values), 'total': sum(values) if values else None,
            'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None}


def analyze(raw, baseline, cases):
    """Validate provenance before scoring; neither extraction sees references."""
    paths = {'raw': Path(raw), 'baseline': Path(baseline), 'cases': Path(cases)}
    records = _load_unique(paths['raw'], 'FaithAct raw')
    initial = _load_unique(paths['baseline'], 'baseline')
    dataset = _load_unique(paths['cases'], 'cases')
    ids, baseline_ids, dataset_ids = set(records), set(initial), set(dataset)
    if baseline_ids - dataset_ids:
        raise ValueError(f'Baseline IDs missing from cases: {_ordered(baseline_ids - dataset_ids)}')
    if ids - baseline_ids:
        raise ValueError(f'FaithAct IDs missing from baseline: {_ordered(ids - baseline_ids)}')
    mismatches = [i for i in _ordered(ids)
                  if not isinstance(records[i].get('original_cot'), str)
                  or records[i]['original_cot'] != raw_answer(initial[i])]
    if mismatches:
        raise ValueError(f'FaithAct original_cot does not exactly match baseline raw text: {mismatches}')

    # Freeze both model decisions before any comparison with the reference.
    left_answer = {i: extract_answer(raw_answer(initial[i]), dataset[i]['question']) for i in ids}
    right_answer = {i: extract_answer(raw_answer(records[i]), dataset[i]['question']) for i in ids}
    references = {i: dataset[i]['reference'] for i in ids}
    left = {i: bool(left_answer[i]) and normalize(left_answer[i]) == normalize(references[i]) for i in ids}
    right = {i: bool(right_answer[i]) and normalize(right_answer[i]) == normalize(references[i]) for i in ids}

    evidence = {}
    noun_calls, extraction_errors = {}, []
    for i, row in records.items():
        entries = row.get('evidence', [])
        if not isinstance(entries, list):
            raise ValueError(f'FaithAct evidence for ID {i} is not a list')
        for entry in entries:
            if (not isinstance(entry, dict) or not isinstance(entry.get('noun'), str)
                    or not isinstance(entry.get('probability'), (float, int))
                    or not math.isfinite(entry['probability'])
                    or not isinstance(entry.get('boxes'), list)):
                raise ValueError(f'Invalid noun inspection record for ID {i}')
        evidence[i] = entries
        # Match FaithAct.run exactly: even an empty interior block invokes the
        # noun extractor. Complete output rows imply these calls returned.
        noun_calls[i] = len(row['original_cot'].split('\n\n')[1:-1])
        errors = row.get('extraction_errors', [])
        if not isinstance(errors, list):
            raise ValueError(f'FaithAct extraction_errors for ID {i} is not a list')
        extraction_errors.extend({'id': i, 'detail': error} for error in errors)

    tool = {i for i in ids if evidence[i]}
    interior = {i for i in ids if noun_calls[i]}
    shown_boxes = {i for i in tool if any(e['probability'] >= .4 and e['boxes'] for e in evidence[i])}
    threshold_text = {i for i in tool if any(e['probability'] >= .4 for e in evidence[i])}
    groups = {
        'all': ids, 'tool_records': tool, 'no_tool_records': ids - tool,
        'boxes_in_evidence_prompt': shown_boxes,
        'tool_records_but_no_boxes_in_prompt': tool - shown_boxes,
        'noun_prefixes_only': tool - threshold_text,
        'interior_but_no_tool_records': interior - tool,
    }
    unresolved_mcq = {i for i in ids if (choices := choices_from_question(dataset[i]['question']))
                      and right_answer[i] not in choices}
    paired_groups = {}
    for name, members in groups.items():
        pair = paired_outcomes(left, right, members)
        pair['baseline_accuracy'] = pair['left_correct'] / pair['n'] if pair['n'] else None
        pair['faithact_accuracy'] = pair['right_correct'] / pair['n'] if pair['n'] else None
        pair['faithact_unresolved_mcq'] = len(members & unresolved_mcq)
        pair['seconds'] = {
            'baseline_total': _timing(initial, members, 'elapsed_seconds'),
            'faithact_total': _timing(records, members, 'elapsed_seconds'),
            'faithact_incremental': _timing(records, members, 'incremental_elapsed_seconds'),
        }
        paired_groups[name] = pair

    nouns = [entry for entries in evidence.values() for entry in entries]
    bands = Counter('exists_gt_0.6' if entry['probability'] > .6 else
                    'maybe_0.4_to_0.6' if entry['probability'] >= .4 else 'below_0.4'
                    for entry in nouns)
    changed = []
    for i in _ordered(ids):
        if left[i] == right[i]:
            continue
        changed.append({
            'id': i, 'direction': 'correction' if right[i] else 'regression',
            'reference': references[i], 'cot_answer': left_answer[i], 'faithact_answer': right_answer[i],
            'question': dataset[i]['question'], 'cot_raw': raw_answer(initial[i]),
            'faithact_raw': raw_answer(records[i]), 'tool_called': i in tool,
            'boxes_in_prompt': i in shown_boxes,
            'nouns': [{'noun': e['noun'], 'probability': e['probability'],
                       'detected_box_count': len(e['boxes']),
                       'shown_box_count': len(e['boxes']) if e['probability'] >= .4 else 0}
                      for e in evidence[i]],
        })

    return {
        'schema_version': 'faithact_diagnostics_v1', 'parser_version': PARSER_VERSION,
        'parser_sha256': _sha(inspect.getsourcefile(extract_answer)),
        'reference_blind_extraction': True,
        'n': len(ids), 'completed_n': len(ids), 'expected_n': len(dataset_ids),
        'baseline_n': len(baseline_ids),
        'is_partial': ids != dataset_ids,
        'is_partial_scope': 'cases file; a complete development subset can still be partial here',
        'is_partial_baseline_subset': ids != baseline_ids,
        'missing_baseline_case_ids': _ordered(baseline_ids - ids),
        'missing_dataset_case_ids': _ordered(dataset_ids - ids),
        'interpretation': ('Paired descriptive comparison on exactly matched initial CoT text; '
                           'tool subsets do not isolate causal contributions or establish matched compute.'),
        'sources': {role: {'path': str(path), 'sha256': _sha(path)} for role, path in paths.items()},
        'original_cot_check': {'checked_n': len(ids), 'exact_match': True, 'mismatch_ids': []},
        'status_counts': dict(Counter(row.get('status', '') for row in records.values())),
        'coverage': {
            'interior_cases': len(interior), 'noun_extraction_calls': sum(noun_calls.values()),
            'noun_calls_by_case': {i: noun_calls[i] for i in _ordered(ids)},
            'noun_call_count_source': "len(original_cot.split('\\n\\n')[1:-1]); released execution loop",
            'tool_cases': len(tool), 'noun_inspections': len(nouns),
            'cases_with_boxes_in_prompt': len(shown_boxes),
            'noun_records_with_any_detected_boxes': sum(bool(e['boxes']) for e in nouns),
            'noun_records_with_boxes_in_prompt': sum(e['probability'] >= .4 and bool(e['boxes']) for e in nouns),
            'detected_box_count': sum(len(e['boxes']) for e in nouns),
            'shown_box_count': sum(len(e['boxes']) for e in nouns if e['probability'] >= .4),
            'noun_probability_bands': {key: bands[key] for key in
                                       ('below_0.4', 'maybe_0.4_to_0.6', 'exists_gt_0.6')},
            'top_nouns': Counter(e['noun'] for e in nouns).most_common(20),
        },
        'noun_extraction': {
            'cases_with_errors': len({error['id'] for error in extraction_errors}),
            'error_count': len(extraction_errors), 'errors': extraction_errors,
            'accepted_format': 'ast.literal_eval(raw.strip()) -> list[str]',
            'successful_raw_format_logged': False,
            'note': 'No subtype is inferred for successful noun calls whose raw text was not logged.',
        },
        'paired_groups': paired_groups,
        'regressions_to_unresolved_ids': _ordered(set(paired_groups['all']['losses']) & unresolved_mcq),
        'changed_correctness': changed,
        'timing_note': ('FaithAct total includes cached initial-CoT cost when recorded by run.py; '
                        'incremental time measures this FaithAct execution. Missing timings are not zeros.'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve() in {path.resolve() for path in (args.raw, args.baseline, args.cases)}:
        parser.error('--output must not overwrite an input file')
    report = analyze(args.raw, args.baseline, args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'n': report['n'], 'is_partial': report['is_partial'],
                      'is_partial_baseline_subset': report['is_partial_baseline_subset'],
                      'parser_version': report['parser_version'], 'all': report['paired_groups']['all'],
                      'coverage': {key: value for key, value in report['coverage'].items()
                                   if key not in ('noun_calls_by_case', 'top_nouns')}},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
