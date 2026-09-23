"""Failure-only, candidate-blind evidence rereading for the frozen online model.

The reread is a new proposal source, not another verifier or an accepted premise.
It enters the same audit and bounded repair controller as every other candidate.
"""
from dataclasses import replace
from copy import deepcopy
import json
import math
import re

from .answers import extract_answer
from .core import Step, parse_json
from .online import Online, PROPOSE, STEP_SCHEMA, question_stem, sanitize_evidence_question
from .prompts import answer_format, task_question


READ_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'observation': {'type': 'string', 'minLength': 1, 'maxLength': 600},
        'remaining_question': {'type': 'string', 'maxLength': 200},
        'evidence_complete': {'type': 'boolean'},
    },
    'required': ['observation', 'remaining_question', 'evidence_complete'],
}

ATOMIC_READ_SCHEMA = deepcopy(READ_SCHEMA)
ATOMIC_READ_SCHEMA['properties']['observation']['maxLength'] = 180
ATOMIC_STEP_SCHEMA = deepcopy(STEP_SCHEMA)
ATOMIC_STEP_SCHEMA['properties']['claim']['maxLength'] = 140


def claim_format_diagnostics(claim, limit):
    """Record lexical truncation signals without treating them as an audit."""
    signals = []
    if len(claim) >= limit - 2:
        signals.append('near_schema_length_limit')
    if re.search(r'\b(?:and|or|but|because|with|of|from|to|the|a|an|its|their|including|provided)'
                 r'[\s,;:\-]*$', claim, flags=re.I):
        signals.append('unfinished_clause_ending')
    return {'characters': len(claim), 'schema_max_characters': limit,
            'truncation_suspected': bool(signals), 'truncation_signals': signals}


class OnlineRepair(Online):
    """Change evidence acquisition after failure instead of reiterating a claim."""

    def __init__(self, *args, repair_style='reread', **kwargs):
        if repair_style not in ('reread', 'atomic_v1'):
            raise ValueError(f'Unknown repair style: {repair_style}')
        super().__init__(*args, **kwargs)
        self.repair_style = repair_style
        self.pending_readback = None
        self.audited_candidates = []
        self.last_finalization = None

    def propose(self, image, question, ledger, feedback):
        if not ledger and not feedback:
            self.audited_candidates = []
            self.last_finalization = None
        redirect = feedback.startswith('Choose a different evidence point')
        if not feedback or redirect or self.last_candidate is None:
            self.pending_readback = None
            return super().propose(image, question, ledger, feedback)

        self.repair_round += 1
        round_index = min(self.repair_round, 3)
        stem = question_stem(question)
        query = stem
        if round_index == 2 and self.pending_readback:
            query = self.pending_readback['remaining_question'].strip() or stem
        # No rejected candidate, proposed option, or assertion-bearing audit
        # feedback is sent to the acquisition or rereading call.
        view, box = image, ()
        if round_index < 3:
            acquisition = (
                'Find the visible evidence needed to inspect this question: ' + query +
                ' Include all queried objects for counting, both referenced objects for '
                'a spatial relation, and distinguishing visible parts for orientation. '
                'Return the smallest region containing this evidence, not an answer.')
            box = self.locate(image, acquisition)
            # Keep a small context margin rather than growing the same region
            # until it becomes a duplicate of the full scene.
            dx, dy = (box[2] - box[0]) * .08, (box[3] - box[1]) * .08
            box = (max(0., box[0] - dx), max(0., box[1] - dy),
                   min(1., box[2] + dx), min(1., box[3] + dy))
            view = [image, self.crop(image, box)]

        read_prompt = (
            'Inspect the visual evidence afresh. Report observations, not an option letter '
            'or a guessed final answer. No earlier answer is available in this inspection. '
            'For a count, enumerate physically distinct visible instances and distinguish '
            'the requested objects from similar background objects. For orientation, '
            'describe visible front/rear or head/tail cues relative to the original image '
            'frame. For a relation, inspect both referenced objects. Do not infer motion, '
            'hidden objects, or metric distances from an unsupported assumption. '
            'If a needed detail is unclear, say what is visible and what remains unclear. '
            'Return JSON with observation (brief directly visible features), '
            'remaining_question (a neutral question about the missing visible cue, '
            'without proposing its value, or an empty string), and evidence_complete '
            '(whether all required visible cues can be inspected). '
            f'Visual question: {stem}\nInspection focus: {query}\n')
        if box:
            read_prompt += ('Image 1 is the complete scene. Image 2 magnifies the selected '
                            'region. Coordinates and left/right always refer to Image 1. '
                            'Check the complete scene for additional instances outside the crop.')
        else:
            read_prompt += ('Use this complete image to inspect global coverage and any '
                            'context that a local region could miss.')
        if self.repair_style == 'atomic_v1':
            read_prompt += (
                '\nKeep observation within 180 characters. Answer ONLY the current neutral '
                'Inspection focus with one complete, concise visible observation. '
                'Do not provide a general scene description, background details, or a '
                'chain of explanations. If the requested cue cannot be resolved, '
                'state that specific ambiguity briefly.')
        read_schema = ATOMIC_READ_SCHEMA if self.repair_style == 'atomic_v1' else READ_SCHEMA
        readback = parse_json(self.vlm.generate(view, read_prompt, max_tokens=320,
                                               schema=read_schema))
        if (not isinstance(readback, dict) or
                not isinstance(readback.get('observation'), str) or
                not readback['observation'].strip() or
                not isinstance(readback.get('remaining_question'), str) or
                not isinstance(readback.get('evidence_complete'), bool)):
            raise ValueError('Independent reread must return a visible observation and missing-cue question')
        self.pending_readback = dict(readback, repair_round=round_index,
                                     inspection_query=query, inspection_box=list(box),
                                     view='full_plus_crop' if box else 'full')

        repair_instruction = (
            'A fresh inspection produced the following TENTATIVE observations. They are '
            'not verified premises and must pass the next audit. Form one corrected atomic '
            'step from the directly visible detail, or state the unresolved ambiguity. '
            'Do not convert evidence_complete into a verification label. '
            'Separate an observed cue from any inference based on it.\n' +
            json.dumps(readback, ensure_ascii=False))
        prompt = PROPOSE % (task_question(question), self.ledger_text(ledger), repair_instruction)
        prompt += '\nFor final_answer only: ' + answer_format(question)
        if box:
            prompt += '\nImage 1 is the full scene; Image 2 is the inspection crop.'
        if self.repair_style == 'atomic_v1':
            prompt += (
                '\nFor this repair, claim must be one complete, directly checkable atomic '
                'observation within 140 characters, answering the current neutral evidence '
                f'question: {query}\n'
                'Extract only the relevant visible fact from the tentative readback. '
                'Do not copy a multi-clause reading into claim. Do not add scene background, '
                'justification, or a reasoning chain. Put an answer only in final_answer. '
                'A short claim is still tentative and must pass the ordinary audit.')
        step_schema = ATOMIC_STEP_SCHEMA if self.repair_style == 'atomic_v1' else STEP_SCHEMA
        value = parse_json(self.vlm.generate(view, prompt, max_tokens=256, schema=step_schema))
        query = value.get('evidence_question', '')
        if '?' in query:
            value['evidence_question'] = query.split('?', 1)[0] + '?'
        step = sanitize_evidence_question(replace(Step.from_dict(value), box=box), question)
        if re.fullmatch(r'[A-Ea-e]|[Yy]es|[Nn]o|[\d.]+', step.claim.strip(' .')):
            raise ValueError('State a complete visual fact, not a bare answer label')
        self.last_candidate = step
        return step

    def audit(self, image, question, step, ledger):
        result = super().audit(image, question, step, ledger)
        if self.pending_readback is not None:
            result.scores['repair_readback'] = dict(self.pending_readback)
        if self.repair_style == 'atomic_v1':
            limit = (ATOMIC_STEP_SCHEMA if self.pending_readback is not None else STEP_SCHEMA
                     )['properties']['claim']['maxLength']
            result.scores['repair_style'] = self.repair_style
            result.scores['claim_format'] = claim_format_diagnostics(step.claim, limit)
        support = result.scores.get('support')
        answer = (step.final_answer or '').strip()
        labels = re.findall(r'(?m)^\s*([A-E])[.)]\s', question)
        normalized = extract_answer(answer, question) if labels else answer
        valid = (normalized in labels if labels else
                 bool(answer) and answer.lower() not in ('null', 'none', 'unknown'))
        if (step.kind == 'visual' and valid and isinstance(support, (float, int))
                and math.isfinite(support)):
            self.audited_candidates.append({
                'answer': normalized,
                'support': float(support), 'claim': step.claim,
                'candidate_index': len(self.audited_candidates),
                'audit_status': result.status,
            })
        return result

    def finalize(self, image, question, ledger):
        if not any(step.verification == 'verified' for step in ledger) and self.audited_candidates:
            # All these candidates remain tentative. Inconclusive inspection
            # need not restore an older, worse-supported hypothesis. The score
            # is model support, not calibrated correctness; exact ties select
            # the latest candidate deterministically.
            selected = max(self.audited_candidates,
                           key=lambda candidate: (candidate['support'], candidate['candidate_index']))
            self.last_finalization = dict(selected, source='highest_support_tentative_candidate',
                                          verification='uncertain', tie_break='latest_candidate')
            return selected['answer']
        self.last_finalization = {
            'source': ('verified_ledger_synthesis' if any(step.verification == 'verified' for step in ledger)
                       else 'parent_fallback'),
            'verification': 'uncertain',
        }
        return super().finalize(image, question, ledger)
