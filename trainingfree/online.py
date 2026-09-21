"""Stepwise support gate plus optional localized deletion controls (MVP)."""
from dataclasses import asdict
import json
import random

from .core import Audit, Step, parse_json

STEP_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {'claim': {'type':'string', 'minLength':1},
                   'kind': {'enum':['visual','deduction']},
                   'evidence_question': {'type':'string'},
                   'box': {'type':'array','items':{'type':'number'},'maxItems':4},
                   'final_answer': {'type':['string','null']}},
    'required': ['claim','kind','evidence_question','box','final_answer']}
BOX_SCHEMA = {'type':'object', 'additionalProperties':False,
              'properties': {'box': {'type':'array','items':{'type':'number'},'minItems':4,'maxItems':4}},
              'required':['box']}

PROPOSE = '''Solve the visual question incrementally. Return EXACTLY ONE atomic next
step, not a full reasoning chain. Treat only the accepted steps as premises.
A visual step must state a visible fact; a deduction may only combine accepted
facts and ordinary logic, and must not introduce any new visual fact.
When sufficient evidence exists, return a deduction with the final concise answer
(just the letter for multiple choice); otherwise final_answer must be null.
Return JSON with kind set to exactly "visual" or exactly "deduction":
{"claim":"one fact or deduction", "kind":"visual",
"evidence_question":"neutral description of which region to inspect",
"box":[x0,y0,x1,y1], "final_answer":null}.
Box coordinates are normalized to [0,1]; deduction boxes are [].
Question: %s
Accepted steps: %s
Repair instruction (not an accepted premise): %s'''


def pixel_box(box, size):
    w, h = size
    return (int(box[0]*w), int(box[1]*h), min(w, max(int(box[2]*w), int(box[0]*w)+1)),
            min(h, max(int(box[3]*h), int(box[1]*h)+1)))


def overlap(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return intersection / max(1e-9, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection)


def control_boxes(box, count=2, seed=0):
    rng = random.Random(seed)
    width, height = box[2]-box[0], box[3]-box[1]
    controls = []
    for _ in range(200):
        x, y = rng.uniform(0, 1-width), rng.uniform(0, 1-height)
        other = (x, y, x+width, y+height)
        if overlap(box, other) <= .1 and all(overlap(other, c) < .8 for c in controls):
            controls.append(other)
            if len(controls) == count:
                break
    return controls


def mask(image, box):
    from PIL import ImageDraw, ImageStat
    result = image.copy()
    neutral = tuple(round(v) for v in ImageStat.Stat(image).mean[:3])
    x0, y0, x1, y1 = pixel_box(box, image.size)
    ImageDraw.Draw(result).rectangle((x0, y0, x1-1, y1-1), fill=neutral)
    return result


class Online:
    def __init__(self, vlm, counterfactual=False, support=.8, specificity=.05):
        self.vlm, self.counterfactual = vlm, counterfactual
        self.support, self.specificity = support, specificity

    def propose(self, image, question, ledger, feedback):
        prompt = PROPOSE % (question, json.dumps([asdict(s) for s in ledger]), feedback)
        return Step.from_dict(parse_json(self.vlm.generate(image, prompt, schema=STEP_SCHEMA)))

    def audit(self, image, question, step, ledger):
        if step.kind == 'deduction':
            prompt = ('Check the inference using ONLY the accepted premises below. '
                      'Reject any new visual fact not entailed by them. Is the entire proposed '
                      'claim justified, and, if supplied, does its final answer correctly answer '
                      'the question from these premises? Reply only Yes or No.\n'
                      f'Question: {question}\nPremises: {json.dumps([s.claim for s in ledger])}\n'
                      f'Claim: {step.claim}\nFinal answer: {step.final_answer}')
            score = self.vlm.yes_probability(None, prompt)
            return Audit('supported' if score >= self.support else 'inconclusive',
                         {'entailment': score}, 'Deduction entailment gate')
        if not step.box:
            return Audit('inconclusive', reason='Missing evidence box; localize and retry.')
        # Final answers are only emitted by a separately checked deduction.
        if step.final_answer is not None:
            return Audit('inconclusive', reason='First verify the visual fact, then derive the final answer.')
        prompt = ('Is the following visual claim fully supported by the image? '
                  'Missing, ambiguous, partially true, or unsupported evidence means No. '
                  'Reply only Yes or No.\nClaim: ' + step.claim)
        original = self.vlm.yes_probability(image, prompt)
        scores = {'support': original, 'box': step.box}
        if original < self.support:
            return Audit('contradicted' if original < 1-self.support else 'inconclusive',
                         scores, 'Insufficient original-image support; reinspect or correct the claim.')
        if not self.counterfactual:
            return Audit('supported', scores, 'Original-image support gate')
        # Reacquire the region in an independent context without the proposed answer.
        route_prompt = ('Locate the smallest region containing all evidence needed to inspect: '
                        + step.evidence_question + '. Return only JSON {"box":[x0,y0,x1,y1]}, '
                        'normalized [0,1]. Do not answer the question.')
        if not step.evidence_question.strip():
            return Audit('inconclusive', scores, 'Need a neutral evidence question for localization.')
        try:
            routed = parse_json(self.vlm.generate(image, route_prompt, max_tokens=100, schema=BOX_SCHEMA))
            box = Step.from_dict({'claim': 'localization', 'box': routed['box']}).box
            if not box:
                raise ValueError('Empty routed box')
        except (ValueError, TypeError, KeyError):
            return Audit('inconclusive', scores, 'Independent localization failed; rebox.')
        scores['routed_box'] = box
        controls = control_boxes(box)
        if len(controls) < 2:
            return Audit('inconclusive', scores, 'Region too broad for matched deletion controls; localize more precisely.')
        removed = self.vlm.yes_probability(mask(image, box), prompt)
        control_scores = [self.vlm.yes_probability(mask(image, b), prompt) for b in controls]
        specific = (original - removed) - sum(original-c for c in control_scores)/len(controls)
        scores.update(target_mask=removed, control_scores=control_scores,
                      control_boxes=controls, specificity=specific)
        if specific < self.specificity:
            return Audit('inconclusive', scores,
                         'Localized counterfactual dependence is weak; reacquire evidence or revise.')
        return Audit('supported', scores, 'Original support and localized counterfactual specificity passed')
