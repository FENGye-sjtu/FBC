"""No speculative text is committed before its audit passes."""
from dataclasses import asdict, dataclass, field
import json
import math
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


def solve(backend: Backend, image, question: str, max_steps=6, repairs=2):
    if max_steps < 1 or repairs < 0:
        raise ValueError('Invalid controller budget')
    ledger, trace = [], []
    for index in range(max_steps):
        feedback = ''
        for attempt in range(repairs + 1):
            try:
                step = backend.propose(image, question, list(ledger), feedback)
            except (ValueError, TypeError, KeyError) as exc:
                feedback = f'Invalid step format: {exc}. Return one valid JSON step.'
                trace.append({'step': index, 'attempt': attempt, 'parse_error': str(exc)})
                continue
            if step.kind == 'deduction' and not ledger:
                audit = Audit('inconclusive', reason='A deduction requires verified premises.')
            else:
                audit = backend.audit(image, question, step, list(ledger))
            trace.append({'step': index, 'attempt': attempt, 'candidate': asdict(step),
                          'parents': list(range(len(ledger))), 'audit': asdict(audit)})
            if audit.status == 'supported':
                ledger.append(step)
                if step.final_answer is not None:
                    return {'answer': step.final_answer, 'status': 'verified_by_model',
                            'ledger': [asdict(s) for s in ledger], 'trace': trace}
                break
            # This is ephemeral repair context, never a premise of a later step.
            feedback = (f'Previous tentative claim {step.claim!r} was not accepted: '
                        f'{audit.reason}. Reinspect the image and replace this step. '
                        'Do not continue from the rejected claim.')
        else:
            return {'answer': None, 'status': 'verification_exhausted',
                    'ledger': [asdict(s) for s in ledger], 'trace': trace}
    return {'answer': None, 'status': 'step_budget_exhausted',
            'ledger': [asdict(s) for s in ledger], 'trace': trace}
