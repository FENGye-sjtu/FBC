"""Read-only diagnostics for completed rows of online reasoning traces."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from .answers import PARSER_VERSION, answer_source, choices_from_question, extract_answer, raw_answer
from .run import normalize


def read_rows(path):
    snapshot = Path(path).read_bytes()
    rows = [json.loads(line) for line in snapshot.decode('utf-8').splitlines() if line.strip()]
    ids = [str(row['id']) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate IDs in {path}')
    return rows, hashlib.sha256(snapshot).hexdigest()


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean(values):
    values = [value for value in values if numeric(value)]
    return statistics.mean(values) if values else None


def first_attempt(trace):
    # A failed initial JSON parse is still the initial attempt. Do not replace
    # it with a later successful repair and inflate the apparent starting score.
    for event in trace:
        if 'attempt' in event and ('candidate' in event or 'parse_error' in event):
            return event
    for event in trace:
        if 'candidate' in event and event.get('event') != 'uncertainty_commit':
            return event
    return {}


def score_text(value, question, reference):
    """The reference enters only exact-match scoring, never extraction."""
    if value is None or not str(value).strip():
        return None, None
    answer = extract_answer(str(value), question)
    if not answer or normalize(answer) in ('none', 'null', 'unknown', 'uncertain', 'undetermined', 'n/a'):
        return answer, None
    choices = choices_from_question(question)
    if choices and answer not in choices:
        return answer, None
    if reference is None:
        return answer, None
    return answer, normalize(answer) == normalize(reference)


def diagnose_row(row, case=None):
    has_public_case = case is not None
    case = case or {}
    case_id = str(row['id'])
    if ('reference' in case and 'reference' in row and
            normalize(case['reference']) != normalize(row['reference'])):
        raise ValueError(f'Reference mismatch for ID {case_id}')
    question = case.get('question', row.get('question', ''))
    reference = case.get('reference', row.get('reference'))
    trace = row.get('trace') or []
    initial_event = first_attempt(trace)
    initial_candidate = initial_event.get('candidate') or {}
    initial_raw = initial_candidate.get('final_answer')
    initial_answer, initial_correct = score_text(initial_raw, question, reference)
    initial_unavailable_reason = None
    if not question:
        # Without public task/options, an output such as "N" cannot be treated
        # as a valid initial MCQ choice merely because a final stored score exists.
        initial_correct = None
        initial_unavailable_reason = 'public_question_unavailable'
    elif reference is None:
        initial_unavailable_reason = 'reference_unavailable'
    elif initial_correct is None:
        initial_unavailable_reason = 'missing_or_unresolved_initial_answer'
    value = raw_answer(row)
    final_answer, rescored_correct = score_text(value, question, reference)
    if rescored_correct is None and reference is not None:
        rescored_correct = False
    stored_correct = row.get('correct') if isinstance(row.get('correct'), bool) else None
    # Public cases provide the common task and reference needed to re-score all
    # runs identically. Historical booleans remain a separate diagnostic.
    final_correct = (rescored_correct if has_public_case else
                     stored_correct if stored_correct is not None else rescored_correct)
    if final_correct is None and reference is not None:
        final_correct = False
    final_source = ('reference_rescore' if has_public_case or stored_correct is None else 'stored_correct')
    audits = [event for event in trace if isinstance(event.get('audit'), dict)]
    repair_attempts = {(event.get('step'),event.get('attempt')) for event in trace
                       if numeric(event.get('attempt')) and event['attempt'] >= 1 and
                       ('audit' in event or 'parse_error' in event)}
    support, readbacks, target_masks, control_masks = [], [], 0, 0
    answer_withheld, option_checks = 0, 0
    for event in audits:
        audit = event['audit']
        scores = audit.get('scores') or {}
        if numeric(scores.get('support')):
            support.append({'step':event.get('step'), 'attempt':event.get('attempt'),
                            'value':scores['support'], 'audit_status':audit.get('status')})
        answer_withheld += bool(scores.get('answer_withheld'))
        option_checks += 'option_counterfactual_scores' in scores
        if 'target_mask' in scores:
            target_masks += 1
            control_masks += len(scores.get('control_scores') or [])
        if isinstance(scores.get('repair_readback'), dict):
            readbacks.append({'step':event.get('step'), 'attempt':event.get('attempt'),
                              **scores['repair_readback']})
    event_counts = Counter(event.get('event') for event in trace if event.get('event'))
    repeated = {event: count for event,count in event_counts.items() if event.startswith('repeat_of_')}
    finalization = row.get('finalization') or {}
    if finalization.get('source'):
        finalization_source = finalization['source']
    elif event_counts['budget_finalization'] or event_counts['redirection_exhausted']:
        finalization_source = 'unrecorded_fallback'
    elif (row.get('ledger') or [{}])[-1].get('final_answer') is not None:
        finalization_source = 'accepted_step'
    else:
        finalization_source = 'not_recorded'
    comparable = initial_correct is not None and final_correct is not None
    return {
        'id':case_id, 'status':row.get('status',''), 'reference':reference,
        'initial_raw_answer':initial_raw, 'initial_answer':initial_answer,
        'initial_answer_missing': initial_correct is None, 'initial_correct':initial_correct,
        'initial_unavailable_reason':initial_unavailable_reason,
        'initial_attempt':initial_event.get('attempt'), 'initial_step':initial_event.get('step'),
        'final_answer':final_answer, 'final_correct':final_correct, 'final_score_source':final_source,
        'answer_source':answer_source(row), 'stored_correct':stored_correct,
        'final_rescored_correct':rescored_correct,
        'stored_rescore_disagreement': (isinstance(row.get('correct'),bool) and
                                       rescored_correct is not None and row['correct'] != rescored_correct),
        'paired_initial_final':comparable,
        'gain':bool(comparable and final_correct and not initial_correct),
        'loss':bool(comparable and initial_correct and not final_correct),
        'audits':len(audits), 'audit_status_counts':dict(Counter(event['audit'].get('status','') for event in audits)),
        'repair_attempts':len(repair_attempts), 'has_repair':bool(repair_attempts),
        'repair_audits':sum(numeric(event.get('attempt')) and event['attempt'] >= 1 for event in audits),
        'parse_errors':sum('parse_error' in event for event in trace),
        'uncertainty_commits':event_counts['uncertainty_commit'],
        'cf_checks':target_masks, 'cf_target_masks':target_masks,
        'cf_control_masks':control_masks, 'cf_mask_calls':target_masks+control_masks,
        'option_consistency_checks':option_checks,
        'repeated_question_events':event_counts['repeat_of_exhausted_question'],
        'repeated_events':repeated, 'repeated_events_total':sum(repeated.values()),
        'support_scores':support, 'mean_support':mean(item['value'] for item in support),
        'answer_withheld':answer_withheld,
        'logged_readback_n':len(readbacks), 'readbacks':readbacks,
        'finalization_source':finalization_source, 'finalization':finalization,
        'elapsed_seconds':row.get('elapsed_seconds'), 'calls':row.get('calls')}


def analyze(path, cases_path=None):
    rows, raw_hash = read_rows(path)
    cases, dataset_hash = {}, None
    if cases_path:
        case_rows, dataset_hash = read_rows(cases_path)
        cases = {str(row['id']):row for row in case_rows}
        missing = [str(row['id']) for row in rows if str(row['id']) not in cases]
        if missing:
            raise ValueError(f'Cases file missing prediction IDs: {missing[:10]}')
    details = [diagnose_row(row,cases.get(str(row['id']))) for row in rows]
    known_final = [row for row in details if row['final_correct'] is not None]
    known_stored = [row for row in details if row['stored_correct'] is not None]
    initial = [row for row in details if row['initial_correct'] is not None]
    paired = [row for row in details if row['paired_initial_final']]
    statuses = Counter(row['status'] for row in details)
    total_fields = ('audits','repair_attempts','repair_audits','parse_errors','uncertainty_commits',
                    'cf_checks','cf_target_masks','cf_control_masks','cf_mask_calls',
                    'option_consistency_checks','repeated_question_events','repeated_events_total',
                    'answer_withheld','logged_readback_n')
    support = [score['value'] for row in details for score in row['support_scores']]
    summary = {
        'path':str(path), 'raw_sha256':raw_hash, 'dataset_sha256':dataset_hash,
        'parser_version':PARSER_VERSION,
        'completed_n':len(details), 'scored_n':len(known_final),
        'correct':sum(row['final_correct'] for row in known_final),
        'accuracy':mean(int(row['final_correct']) for row in known_final),
        'stored_score_diagnostic':{
            'n':len(known_stored), 'correct':sum(row['stored_correct'] for row in known_stored),
            'accuracy':mean(int(row['stored_correct']) for row in known_stored)},
        'final_score_source_counts':dict(Counter(row['final_score_source'] for row in details)),
        'uncertain_n':statuses['answered_with_uncertainty'],
        'verified_n':statuses['verified_by_model'],
        'error_n':sum(count for status,count in statuses.items() if 'error' in status.lower()),
        'status_counts':dict(statuses),
        'initial_answer_n':len(initial), 'initial_missing_n':len(details)-len(initial),
        'initial_correct':sum(row['initial_correct'] for row in initial),
        'initial_accuracy':mean(int(row['initial_correct']) for row in initial),
        'paired_n':len(paired), 'paired_initial_correct':sum(row['initial_correct'] for row in paired),
        'paired_final_correct':sum(row['final_correct'] for row in paired),
        'gains_ids':[row['id'] for row in paired if row['gain']],
        'losses_ids':[row['id'] for row in paired if row['loss']],
        'initial_missing_ids':[row['id'] for row in details if row['initial_answer_missing']],
        'stored_rescore_disagreement_ids':[row['id'] for row in details if row['stored_rescore_disagreement']],
        'with_repairs_n':sum(row['has_repair'] for row in details),
        'mean_support':mean(support), 'min_support':min(support) if support else None,
        'max_support':max(support) if support else None,
        'finalization_source_counts':dict(Counter(row['finalization_source'] for row in details)),
        'mean_latency_seconds':mean(row['elapsed_seconds'] for row in details),
        'mean_calls':mean(row['calls'] for row in details),
        'trace_count_note':'Counts cover recorded trace events; failed or unlogged model calls cannot be reconstructed.',
        'final_accuracy_protocol':('Re-score the shared source text with public cases and the current parser; '
                                   'saved booleans are diagnostic only.' if cases_path else
                                   'No cases supplied: use saved boolean correct when present, otherwise '
                                   'reference-blind extraction and row reference; see per-case score source.'),
        'reference_blind_extraction':True}
    summary.update({field:sum(row[field] for row in details) for field in total_fields})
    return {'summary':summary, 'details':details}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--cases', type=Path, help='Optional public questions/references for exact option-text parsing')
    parser.add_argument('--output', type=Path, help='Optional full JSON report including per-case diagnostics')
    args = parser.parse_args(argv)
    result = analyze(args.raw,args.cases)
    if args.output:
        protected = {args.raw.resolve()} | ({args.cases.resolve()} if args.cases else set())
        if args.output.resolve() in protected:
            raise ValueError('Output cannot overwrite an input file')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result['summary'],ensure_ascii=False))


if __name__ == '__main__':
    main()
