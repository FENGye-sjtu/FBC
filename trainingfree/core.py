"""Repair at each boundary; unresolved evidence continues explicitly uncertain."""
from dataclasses import asdict, dataclass, field, replace
import json
import math
import re
from typing import Any, Protocol


def parse_json(text: str):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    return json.loads(text)


@dataclass(frozen=True)
class Step:
    claim: str
    kind: str = 'visual'
    evidence_question: str = ''
    box: tuple[float, ...] = ()
    final_answer: str | None = None
    verification: str = 'candidate'
    uncertainty_reason: str = ''

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError('Expected a step object')
        if not isinstance(value.get('claim'), str) or not value['claim'].strip():
            raise ValueError('Missing atomic claim')
        kind = value.get('kind', 'visual')
        if kind not in ('visual', 'deduction'):
            raise ValueError('Unknown step kind')
        box = tuple(float(x) for x in value.get('box', []))
        if box and (len(box) != 4 or not all(math.isfinite(x) and 0 <= x <= 1 for x in box)
                    or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError('Box must be normalized xyxy with positive area')
        final = value.get('final_answer')
        if final is not None and (not isinstance(final, str) or not final.strip()):
            raise ValueError('Final answer must be null or nonempty text')
        question = value.get('evidence_question', '')
        if not isinstance(question, str):
            raise ValueError('Evidence question must be text')
        return cls(value['claim'].strip(), kind, question, box, final)


@dataclass
class Audit:
    status: str  # supported | contradicted | inconclusive
    scores: dict[str, Any] = field(default_factory=dict)
    reason: str = ''

    def __post_init__(self):
        if self.status not in ('supported', 'contradicted', 'inconclusive'):
            raise ValueError('Unknown audit status')


class Backend(Protocol):
    def propose(self, image, question: str, ledger: list[Step], feedback: str) -> Step: ...
    def audit(self, image, question: str, step: Step, ledger: list[Step]) -> Audit: ...
    def finalize(self, image, question: str, ledger: list[Step]) -> str: ...


def solve(backend: Backend, image, question: str, max_steps=6, repairs=3, on_event=None):
    if max_steps < 1 or not 0 <= repairs <= 3:
        raise ValueError('Invalid controller budget')
    ledger, trace = [], []
    def record(event):
        trace.append(event)
        if on_event is not None:
            on_event(event)
    def finish(reason, event='budget_finalization'):
        answer = backend.finalize(image, question, list(ledger))
        record({'event':event,'verification':'uncertain','reason':reason})
        return {'answer':answer,'status':'answered_with_uncertainty',
                'ledger':[asdict(s) for s in ledger],'trace':trace}
    for index in range(max_steps):
        feedback = ''
        last_step = None
        last_reason = 'No well-formed candidate was produced.'
        for attempt in range(repairs + 1):
            try:
                step = backend.propose(image, question, list(ledger), feedback)
            except (ValueError, TypeError, KeyError) as exc:
                feedback = f'Invalid step format: {exc}. Return one valid JSON step.'
                record({'step': index, 'attempt': attempt, 'parse_error': str(exc)})
                continue
            last_step = step
            if ledger and ledger[-1].verification == 'uncertain':
                prior = ledger[-1].claim.removeprefix('Uncertain whether: ')
                key = lambda text: re.sub(r'\W+',' ',text.lower()).strip()
                if key(prior) == key(step.claim):
                    record({'step':index,'event':'repeat_of_exhausted_point','candidate':asdict(step)})
                    return finish('The same exhausted point was proposed again without new committed evidence; '
                                  'continue to an explicitly uncertain best-effort answer, not another repair cycle.',
                                  'uncertain_answer_synthesis')
            if step.kind == 'deduction' and not ledger:
                audit = Audit('inconclusive', reason='A deduction requires verified premises.')
            else:
                audit = backend.audit(image, question, step, list(ledger))
            record({'step': index, 'attempt': attempt, 'candidate': asdict(step),
                          'parents': list(range(len(ledger))), 'audit': asdict(audit)})
            if audit.status == 'supported':
                # Preserve a verified observation even when its proposed answer
                # does not follow. The conclusion is regenerated at the next step.
                if audit.scores.get('answer_withheld'):
                    step = replace(step, final_answer=None)
                inherited = step.kind == 'deduction' and any(s.verification == 'uncertain' for s in ledger)
                step = replace(step, verification='uncertain' if inherited else 'verified',
                               uncertainty_reason='Depends on unresolved premises.' if inherited else '')
                ledger.append(step)
                if step.final_answer is not None:
                    uncertain = any(s.verification == 'uncertain' for s in ledger)
                    return {'answer': step.final_answer,
                            'status': 'answered_with_uncertainty' if uncertain else 'verified_by_model',
                            'ledger': [asdict(s) for s in ledger], 'trace': trace}
                break
            # This is ephemeral repair context, never a premise of a later step.
            feedback = (f'Previous tentative claim {step.claim!r} was not accepted: '
                        f'{audit.reason}. Reinspect the image and replace this step. '
                        'Do not continue from the rejected claim.')
            last_reason = audit.reason or audit.status
        else:
            # Initial attempt + at most 3 repairs. Do not assert a failed claim as fact.
            unresolved = (f'Uncertain whether: {last_step.claim}' if last_step else
                          'Unresolved evidence point: no valid structured observation was obtained.')
            step = Step(unresolved, kind=last_step.kind if last_step else 'visual',
                        evidence_question=last_step.evidence_question if last_step else '',
                        box=last_step.box if last_step else (), verification='uncertain',
                        uncertainty_reason=f'After {repairs} repair rounds: {last_reason}')
            ledger.append(step)
            record({'step':index, 'event':'uncertainty_commit', 'repair_rounds':repairs,
                          'candidate':asdict(step), 'parents':list(range(len(ledger)-1))})
            # Continue to the NEXT reasoning step with this uncertainty visible.
    return finish('Best-effort answer after bounded step budget; not a passed final audit.')
