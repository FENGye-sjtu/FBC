"""Answer-decoupled, verify-before-commit controller.

The initial direct answer is retained only as a tentative safe fallback.  Every
visual premise is proposed without answer options, audited, and only then made
available to a separate deduction step.  A failed deduction sends control back
to acquisition of a new visual fact instead of rewriting the conclusion around
the same premise.
"""
from copy import deepcopy
from dataclasses import replace
import json
import re

from .answers import extract_answer
from .core import Audit, Step, parse_json
from .online import STEP_SCHEMA, question_stem, sanitize_evidence_question
from .online_repair import OnlineRepair
from .prompts import answer_format, task_question


EVIDENCE_STEP_SCHEMA = deepcopy(STEP_SCHEMA)
EVIDENCE_STEP_SCHEMA['properties']['claim']['maxLength'] = 180
EVIDENCE_STEP_SCHEMA['properties']['kind'] = {'enum': ['visual']}
EVIDENCE_STEP_SCHEMA['properties']['final_answer'] = {'type': 'null'}

DEDUCTION_STEP_SCHEMA = deepcopy(STEP_SCHEMA)
DEDUCTION_STEP_SCHEMA['properties']['kind'] = {'enum': ['deduction']}
DEDUCTION_STEP_SCHEMA['properties']['evidence_question'] = {'enum': ['']}
DEDUCTION_STEP_SCHEMA['properties']['final_answer'] = {
    'type': ['string', 'null'], 'maxLength': 80,
}


class EvidenceFirstOnlineRepair(OnlineRepair):
    """Keep option selection out of evidence acquisition and repair."""

    protocol_version = 'evidence_first_v6_anchor_disagreement_gate'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor_raw = ''
        self.anchor_answer = ''

    def _start_case(self, image, question):
        self.audited_candidates = []
        self.last_finalization = None
        self.pending_readback = None
        self.last_candidate = None
        self.repair_round = 0
        # This call deliberately matches the declared direct baseline prompt.
        # It supplies a no-regression fallback, never a verified premise.
        self.anchor_raw = self.vlm.generate(
            image, question + '\nReturn only the concise answer.')
        self.anchor_answer = extract_answer(self.anchor_raw, question)
        self.fallback_answer = self.anchor_answer

    def _visual_proposal(self, image, question, ledger, reason=''):
        stem = question_stem(question)
        prompt = (
            'Use the Grounding -> Observation order for exactly one atomic visual fact. '
            'First identify the exact relevant object(s) or region(s) in the full scene; then '
            'inspect the visible cue needed by the question. Return only the resulting atomic '
            'observation, not the private grounding process. '
            'This is evidence acquisition, not answer selection: no answer choices or '
            'proposed conclusion are available. State only what is directly visible, '
            'not an inferred option or a full chain. For counting, enumerate distinct visible '
            'instances; for orientation, use visible front/rear or head/tail cues; for a '
            'relation, inspect both referenced objects. The evidence_question must name what '
            'and where to inspect without suggesting its value; the claim must answer exactly '
            'that inspection question. Return one JSON object with kind="visual" and '
            'final_answer=null. Do not repeat a verified or unresolved evidence point.\n'
            f'Question without choices: {stem}\n'
            f'Evidence ledger: {self.ledger_text(ledger)}\n'
            f'Acquisition reason: {reason or "begin with the most decisive visible fact"}'
        )
        raw = self.vlm.generate(image, prompt, max_tokens=256, schema=EVIDENCE_STEP_SCHEMA)
        value = parse_json(raw)
        value['kind'] = 'visual'
        value['final_answer'] = None
        step = sanitize_evidence_question(Step.from_dict(value), stem)
        if re.fullmatch(r'[A-Ea-e]|[Yy]es|[Nn]o|[\d.]+', step.claim.strip(' .')):
            raise ValueError('Evidence must be a complete visible fact, not an answer label')
        self.last_candidate = step
        self.pending_readback = None
        return step

    def _deduction_proposal(self, image, question, ledger):
        prompt = (
            'Derive the answer using ONLY the labeled evidence ledger and ordinary logic. '
            'Verified visual facts may be used as premises. Uncertain entries are unresolved '
            'and must not be silently assumed true. Do not introduce or re-describe any new '
            'visual fact. First decide whether the verified premises are sufficient to distinguish '
            'the options. Return one JSON object with kind="deduction", evidence_question="", '
            'and a short claim explaining the logical status. Set final_answer only when it is '
            'fully entailed by the verified premises. If another visual fact is still needed, set '
            'final_answer=null and say what distinction remains unresolved without guessing it.\n'
            f'Question: {task_question(question)}\n'
            f'Evidence ledger: {self.ledger_text(ledger)}\n'
            'For final_answer only: ' + answer_format(question)
        )
        raw = self.vlm.generate(None, prompt, max_tokens=192, schema=DEDUCTION_STEP_SCHEMA)
        value = parse_json(raw)
        value['kind'] = 'deduction'
        value['evidence_question'] = ''
        step = Step.from_dict(value)
        self.last_candidate = step
        self.pending_readback = None
        return step

    def propose(self, image, question, ledger, feedback):
        if not ledger and not feedback:
            self._start_case(image, question)
            return self._visual_proposal(image, question, ledger)
        if not feedback:
            if ledger and ledger[-1].kind == 'deduction':
                # core.check_final removes an answer that fails independent
                # option consistency while preserving its supported logical
                # content. Do not spend every remaining step repeating that
                # same conclusion; acquire a genuinely new visual premise.
                return self._visual_proposal(
                    image, question, ledger,
                    'the previous deduction was retained but its answer was withheld; '
                    'inspect a new decisive visual fact')
            return self._deduction_proposal(image, question, ledger)
        if self.last_candidate is not None and self.last_candidate.kind == 'deduction':
            # Do not let a rejected conclusion prime the next observation.
            return self._visual_proposal(
                image, question, ledger,
                'the previous deduction was not established; inspect a different missing visual fact')
        # Reuse failure-only crop/reread repair, but remove both options and
        # answer generation from the visual branch.  The returned observation
        # must still pass the ordinary support/CF audit.
        stem = question_stem(question)
        step = super().propose(image, stem, ledger, feedback)
        step = replace(step, kind='visual', final_answer=None)
        self.last_candidate = step
        return step

    def audit(self, image, question, step, ledger):
        negative = bool(re.search(
            r"\b(?:no|not|none|without|zero|absent|absence|cannot|can't|isn't|aren't)\b",
            step.claim, flags=re.I))
        if self.counterfactual and step.kind == 'visual' and negative:
            # Deleting pixels cannot supply the missing positive instance that
            # would falsify an absence/negation claim. Requiring deletion
            # specificity here systematically rejects correct negative facts.
            # Keep the ordinary image support gate and record why the spatial
            # intervention is not identifying; positive claims still require CF.
            self.counterfactual = False
            try:
                result = super().audit(image, question, step, ledger)
            finally:
                self.counterfactual = True
            result.scores['cf_applicability'] = 'not_identifying_for_negative_claim'
            result.scores['cf_skipped'] = True
            return result
        result = super().audit(image, question, step, ledger)
        labels = re.findall(r'(?m)^\s*([A-E])[.)]\s', question)
        proposed = (step.final_answer or '').strip().upper()
        anchor = (self.anchor_answer or '').strip().upper()
        verified_visual_n = sum(
            item.kind == 'visual' and item.verification == 'verified' for item in ledger)
        if (step.kind == 'deduction' and result.status == 'supported'
                and not result.scores.get('answer_withheld')
                and proposed in labels and anchor in labels and proposed != anchor
                and verified_visual_n < 2):
            # One self-verified visual impression is not enough to overwrite a
            # valid direct anchor: proposal, support and option scoring all use
            # the same frozen VLM and can share one error.  Acquire a second,
            # distinct verified fact before allowing the changed answer.
            result.scores['answer_withheld'] = True
            result.scores['anchor_disagreement'] = {
                'anchor': anchor, 'proposed': proposed,
                'verified_visual_n': verified_visual_n, 'required_visual_n': 2,
            }
            return Audit('supported', result.scores,
                         'Changed answer has only one verified visual fact; acquire a second '
                         'independent fact before overriding the direct anchor.')
        return result

    def finalize(self, image, question, ledger):
        labels = re.findall(r'(?m)^\s*([A-E])[.)]\s', question)
        valid = (self.anchor_answer.upper() in labels if labels else bool(self.anchor_answer))
        if valid:
            self.last_finalization = {
                'source': 'tentative_direct_anchor',
                'verification': 'uncertain',
                'answer': self.anchor_answer.upper() if labels else self.anchor_answer,
                'policy': 'preserve_anchor_when_no_deduction_passes',
            }
            return self.last_finalization['answer']
        self.last_finalization = {
            'source': 'ledger_synthesis_without_valid_anchor',
            'verification': 'uncertain',
        }
        return super().finalize(image, question, ledger)
