"""Same student/step interface without source audits or repair calls."""
from dataclasses import asdict
from efa.core import Budget, Claim
from efa.format import STEP_END, step_prompt
from efa.qwen import parse_json
from efa.regions import Box


def run_step_only(backend, question):
    budget=Budget();context=[];trace=[];final=None
    for number in range(1,budget.max_steps+1):
        budget.take_step()
        raw=backend.generate(step_prompt(question,context,number,number==budget.max_steps),
            [Box(0,0,backend.image.width,backend.image.height)],max_tokens=384,step_boundary=True)[0]
        try:
            if not raw.rstrip().endswith(STEP_END):raise ValueError('Missing step boundary')
            parsed=parse_json(raw)
            if 'review_parent' in parsed:raise ValueError('Parent audits disabled in this ablation')
            atoms=parsed['atoms']
            if not 1<=len(atoms)<=budget.max_atoms_per_step or not isinstance(parsed.get('final'),bool):
                raise ValueError('Invalid step coverage')
            parents=tuple(n['key'] for n in context)
            claims=[Claim(a['subject'],a['predicate'],a['value'],a.get('family','attribute'),
                          a.get('kind','visual'),parents) for a in atoms]
            if any(c.kind not in ('visual','premise') or not all(isinstance(v,str) and v.strip()
                for v in (c.subject,c.predicate,c.value)) for c in claims):raise ValueError('Invalid claim')
        except (ValueError,KeyError,TypeError) as error:
            trace.append(dict(step=number,raw=raw,reason=str(error)))
            final=dict(answer=raw,verified=False,reason='invalid_or_unsupported_step')
            break
        # Every earlier value remains explicitly unverified. No reference labels,
        # supplied boxes, source reads or gate outputs enter these prompts.
        context.extend(dict(key=f'step{number}.atom{i}@1',claim=asdict(c),verified=False)
                       for i,c in enumerate(claims))
        trace.append(dict(step=number,raw=raw,atoms=[asdict(c) for c in claims]))
        if parsed['final'] or number==budget.max_steps:
            final=dict(answer=claims[-1].value,verified=False,
                       reason='student_final' if parsed['final'] else 'global_step_budget')
            break
    return dict(question=question,final=final,trace=trace,budget=asdict(budget),cost=list(backend.calls),
                implementation='same_student_numbered_steps_without_audits',calibrated=False)
