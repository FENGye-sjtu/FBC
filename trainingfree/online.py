"""Stepwise support gate plus optional localized deletion controls (MVP)."""
from dataclasses import replace
import json
import random
import re

from .core import Audit, Step, parse_json
from .prompts import task_question, answer_format, compact_initial_prompt

STEP_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {'claim': {'type':'string', 'minLength':12, 'maxLength':300},
                   'kind': {'enum':['visual','deduction']},
                   'evidence_question': {'type':'string', 'maxLength':160},
                   'final_answer': {'type':['string','null'], 'maxLength':80}},
    'required': ['claim','kind','evidence_question','final_answer']}
BOX_SCHEMA = {'type':'object', 'additionalProperties':False,
              'properties': {'box': {'type':'array','items':{'type':'number'},'minItems':4,'maxItems':4}},
              'required':['box']}

PROPOSE = '''Solve the visual question incrementally. Return EXACTLY ONE atomic next
step, not a full reasoning chain. The ledger distinguishes verified and uncertain
points. Only verified points are established facts. Uncertain points are unresolved
hypotheses, not true premises; reason conditionally or use independent evidence.
Move forward after an unresolved point; do not get stuck repeating it forever.
A visual step must state a visible fact; a deduction may only combine accepted
facts and ordinary logic, and must not introduce any new visual fact.
For a visual step, evidence_question MUST be a nonempty neutral question about
the visible evidence being checked, even when that evidence is ambiguous.
For a deduction, set evidence_question to an empty string: it introduces no
new image observation and must state its dependence on uncertain premises.
As soon as the evidence answers the question, include the concise final_answer
(option letter for multiple choice, Yes/No for binary questions, otherwise a short
word or number). A visual claim may include a final_answer; both will be checked.
Do not repeat an already verified observation: move to a conclusion or a NEW fact.
Return one JSON object: claim is your actual observation/inference, kind is
"visual" or "deduction", evidence_question is a neutral region-localization
query without your proposed answer, final_answer is a string or null.
The claim MUST be a complete factual sentence, NEVER an option letter or just
the final answer. The evidence_question must be short and must NOT assert the
answer or include reasoning. Put the answer only in final_answer.
Keep the claim short (one sentence). If an earlier point is unresolved after
repair, do not repeat that same point as a new visual step. Inspect a DIFFERENT
relevant fact, or make a conditional deduction to the best-supported answer.
Question: %s
Evidence ledger (preserve uncertainty labels): %s
Repair instruction (not an accepted premise): %s'''


def question_stem(question):
    """Retain the public task without answer choices or answer-only directives."""
    stem = task_question(question)
    stem = re.split(r'(?im)^\s*(?:Options\s*:|[A-E][.)]\s)', stem, maxsplit=1)[0]
    return re.sub(r'^\s*Question\s*:\s*', '', stem, flags=re.I).strip()


def sanitize_evidence_question(step, question):
    """Give empty visual queries a stable budget key without rejecting the step."""
    query = step.evidence_question.strip()
    # A structured decoder can still append an answer after a syntactically
    # complete question ("What color is it? It is red").  Localization and
    # evidence-budget identity must see only the neutral query, never that
    # answer-bearing suffix.  Apply this at the shared boundary so specialized
    # controllers cannot accidentally bypass the older caller-side cleanup.
    if step.kind == 'visual' and '?' in query:
        query = query.split('?', 1)[0].strip() + '?'
    if step.kind == 'visual' and not query:
        query = question_stem(question) or 'What relevant visual evidence is present?'
    return replace(step, evidence_question=query)


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
    def __init__(self, vlm, counterfactual=False, support=.8, specificity=.05, initial_style='full'):
        if initial_style not in ('full', 'compact_v1'):
            raise ValueError(f'Unknown initial proposal style: {initial_style}')
        self.vlm, self.counterfactual = vlm, counterfactual
        self.support, self.specificity = support, specificity
        self.initial_style = initial_style
        self.last_candidate, self.repair_round = None, 0
        self.fallback_answer = None

    @staticmethod
    def ledger_text(ledger):
        # Internal normalized boxes are NOT a generation target; do not expose
        # them as pixel examples and accidentally renormalize copied coordinates.
        return json.dumps([({'claim':s.claim,'verification':s.verification} if s.verification!='uncertain'
                            else {'unresolved_question':s.evidence_question or 'Unresolved visual evidence for the task',
                                  'verification':'uncertain','reason':s.uncertainty_reason}) for s in ledger])

    def locate(self, image, query):
        width, height = self.vlm.model_image_size(image) if hasattr(self.vlm, 'model_image_size') else image.size
        prompt = ('Locate the image region needed to inspect the following query; do not answer it. '
                  f'Query: {query}\nImage PIXEL dimensions as shown to you: width={width}, height={height}. '
                  'Return JSON with box=[left,top,right,bottom] in PIXELS.')
        value = parse_json(self.vlm.generate(image, prompt, max_tokens=100, schema=BOX_SCHEMA))
        box = tuple(v / (width,height)[i % 2] for i,v in enumerate(value['box']))
        return Step.from_dict({'claim':'localized evidence','box':box}).box

    @staticmethod
    def crop(image, box):
        return image.crop(pixel_box(box, image.size))

    def propose(self, image, question, ledger, feedback):
        box, view = (), image
        redirect = feedback.startswith('Choose a different evidence point')
        if not feedback or redirect:
            self.last_candidate, self.repair_round = None, 0
        if not ledger and not feedback:
            # A new case starts. Keep the first answer only as an uncertainty
            # fallback; it is never entered into the verified ledger.
            self.fallback_answer = None
        elif self.last_candidate is not None:
            self.repair_round += 1
            try:
                box = self.locate(image, self.last_candidate.evidence_question or question)
                pad = (.10, .25, .45)[min(self.repair_round-1, 2)]
                dx, dy = (box[2]-box[0])*pad, (box[3]-box[1])*pad
                box = (max(0,box[0]-dx), max(0,box[1]-dy), min(1,box[2]+dx), min(1,box[3]+dy))
                view = [image, self.crop(image, box)]
            except (ValueError, TypeError, KeyError):
                box = ()
        if self.initial_style == 'compact_v1' and not ledger and not feedback:
            prompt = compact_initial_prompt(question)
        else:
            prompt = PROPOSE % (task_question(question), self.ledger_text(ledger), feedback)
            prompt += '\nFor final_answer only: ' + answer_format(question)
        if box:
            prompt += '\nImage 1 is the full scene. Image 2 is a magnified evidence region; use it to correct the previous step.'
        raw = self.vlm.generate(view, prompt, max_tokens=256, schema=STEP_SCHEMA)
        try:
            value = parse_json(raw)
            # Some real outputs append their answer after the localization question.
            # Do not pass that appended assertion into the independent box call.
            query = value.get('evidence_question','')
            if '?' in query:
                value['evidence_question'] = query.split('?',1)[0]+'?'
            step = sanitize_evidence_question(replace(Step.from_dict(value), box=box), question)
            if re.fullmatch(r'[A-Ea-e]|[Yy]es|[Nn]o|[\d.]+', step.claim.strip(' .')):
                raise ValueError('State a complete visual fact, not a bare answer label')
            if step.claim.lower().strip() in ('one fact or deduction','one fact','your actual observation/inference'):
                raise ValueError('Placeholder text is not an observation')
            self.last_candidate = step
            if not ledger and not feedback and step.final_answer is not None:
                self.fallback_answer = step.final_answer.strip()
            return step
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f'{exc}; raw candidate={raw[:1200]}') from exc

    def finalize(self, image, question, ledger):
        has_verified = any(step.verification == 'verified' for step in ledger)
        is_mcq = bool(re.search(r'(?m)^\s*A[.)]\s', question))
        fallback_valid = (bool(re.fullmatch(r'[A-E]', self.fallback_answer or '', re.I))
                          if is_mcq else bool(self.fallback_answer))
        if fallback_valid and not has_verified:
            # An inconclusive verifier is not counter-evidence. If no later
            # fact was verified, do not let an unverified free-form synthesis
            # overwrite the original candidate; preserve it explicitly under
            # the controller's uncertain status.
            return self.fallback_answer.upper() if is_mcq else self.fallback_answer
        prompt = ('Give the best available concise answer to the question using the image '
                  'and evidence ledger. Prefer verified facts. Uncertain statements are NOT '
                  'established facts; do not silently assume them true. If uncertainty remains, '
                  'choose the best-supported answer; the controller records its uncertainty. '
                  'Return ONLY the option letter for multiple choice, otherwise a word or number.\n'
                  f'Question: {task_question(question)}\nLedger: {self.ledger_text(ledger)}\n'
                  + answer_format(question))
        return self.vlm.generate(image, prompt, max_tokens=64)

    def audit(self, image, question, step, ledger):
        if step.kind == 'deduction':
            prompt = ('Check the inference using ONLY the labeled premises below. '
                      'Uncertain premises may support a conditional inference but are not proven facts. '
                      'Reject any new visual fact not entailed by them. Is the entire proposed '
                      'claim justified, and, if supplied, does its final answer correctly answer '
                      'the question from these premises? Reply only Yes or No.\n'
                      f'Question: {task_question(question)}\nPremises: {self.ledger_text(ledger)}\n'
                      f'Claim: {step.claim}\nFinal answer: {step.final_answer}')
            score = self.vlm.yes_probability(None, prompt)
            scores = {'entailment': score}
            if score < self.support:
                return Audit('inconclusive', scores, 'Deduction entailment gate')
            alternative = self.option_counterfactual(image, question, step, scores)
            return alternative or Audit('supported', scores, 'Deduction entailment gate')
        def view(img):
            return [img, self.crop(img, step.box)] if step.box else img
        prompt = ('Is the following visual claim fully supported by the image? '
                  'Missing, ambiguous, partially true, or unsupported evidence means No. '
                  'Reply only Yes or No.\nClaim: ' + step.claim)
        if step.box:
            prompt += '\nImage 1 is the original scene; image 2 is a magnified region from it.'
        original = self.vlm.yes_probability(view(image), prompt)
        scores = {'support': original, 'box': step.box}
        if original < self.support:
            return Audit('contradicted' if original < 1-self.support else 'inconclusive',
                         scores, 'Insufficient original-image support; reinspect or correct the claim.')
        if not self.counterfactual:
            return self.check_final(image, question, step, ledger, scores, 'Original-image support gate')
        # Reacquire the region in an independent context without the proposed answer.
        if not step.evidence_question.strip():
            return Audit('inconclusive', scores, 'Need a neutral evidence question for localization.')
        try:
            box = self.locate(image, step.evidence_question)
            if not box:
                raise ValueError('Empty routed box')
        except (ValueError, TypeError, KeyError):
            return Audit('inconclusive', scores, 'Independent localization failed; rebox.')
        scores['routed_box'] = box
        controls = control_boxes(box)
        if len(controls) < 2:
            return Audit('inconclusive', scores, 'Region too broad for matched deletion controls; localize more precisely.')
        removed = self.vlm.yes_probability(view(mask(image, box)), prompt)
        control_scores = [self.vlm.yes_probability(view(mask(image, b)), prompt) for b in controls]
        specific = (original - removed) - sum(original-c for c in control_scores)/len(controls)
        scores.update(target_mask=removed, control_scores=control_scores,
                      control_boxes=controls, specificity=specific)
        if specific < self.specificity:
            return Audit('inconclusive', scores,
                         'Localized counterfactual dependence is weak; reacquire evidence or revise.')
        return self.check_final(view(image), question, step, ledger, scores,
                                'Original support and localized counterfactual specificity passed')

    def check_final(self, image, question, step, ledger, scores, reason):
        if step.final_answer is not None:
            prompt = ('Does the proposed final answer correctly follow from the visual fact and '
                      'labeled premises for this question? Check the exact option mapping if present. '
                      'Do not invent additional visual facts. Reply only Yes or No.\n'
                      f'Question: {task_question(question)}\nPremises: {self.ledger_text(ledger)}\n'
                      f'Verified visual fact: {step.claim}\nProposed answer: {step.final_answer}')
            scores['answer_entailment'] = self.vlm.yes_probability(None, prompt)
            if scores['answer_entailment'] < self.support:
                scores['answer_withheld'] = True
                return Audit('supported', scores,
                             'Visual fact supported; withhold proposed answer and derive a corrected conclusion next.')
            alternative = self.option_counterfactual(image, question, step, scores)
            if alternative:
                return alternative
        return Audit('supported', scores, reason)

    def option_counterfactual(self, image, question, step, scores):
        labels = re.findall(r'(?m)^\s*([A-E])[.)]\s', question)
        proposed = (step.final_answer or '').strip().upper()
        if not (len(labels) >= 2 and proposed in labels and
                hasattr(self.vlm, 'log_probability')):
            return None
        # This is independent answer-consistency scoring, NOT an image
        # counterfactual intervention. Keep the historical function/trace keys
        # for compatibility; spatial masking in audit() is the CF component.
        contrast_prompt = (
            'Independently choose the single best option from the image. Do not assume that '
            'the candidate claim or candidate answer is correct. Return only the option letter.\n'
            'Question: ' + task_question(question))
        if hasattr(self.vlm, 'option_log_probabilities'):
            option_scores = self.vlm.option_log_probabilities(image, contrast_prompt, labels)
        else:
            option_scores = {label: self.vlm.log_probability(image, contrast_prompt, label)
                             for label in labels}
        best = max(option_scores, key=option_scores.get)
        scores['option_counterfactual_scores'] = option_scores
        scores['option_counterfactual_best'] = best
        if best != proposed:
            scores['answer_withheld'] = True
            return Audit('supported', scores,
                         'Claim supported, but an alternative option has higher grounded likelihood; '
                         'withhold the answer and derive a corrected conclusion next.')
        return None
