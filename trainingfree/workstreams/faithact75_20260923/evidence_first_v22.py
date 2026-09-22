"""Candidate-blind visual rereading for the Evidence-First controller."""

from __future__ import annotations

from copy import deepcopy
import json
import re

from trainingfree.core import Audit, parse_json
from trainingfree.evidence_first import (
    DEDUCTION_STEP_SCHEMA,
    EVIDENCE_STEP_SCHEMA,
    EvidenceFirstOnlineRepair,
)


BLIND_READ_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "observation": {"type": "string", "minLength": 1, "maxLength": 180},
        "resolved": {"type": "boolean"},
    },
    "required": ["observation", "resolved"],
}

V22_DEDUCTION_SCHEMA = deepcopy(DEDUCTION_STEP_SCHEMA)
V22_DEDUCTION_SCHEMA["properties"]["claim"]["maxLength"] = 180

INFERENCE_LANGUAGE = re.compile(
    r"\b(?:because|therefore|thus|hence|indicat(?:e|es|ed|ing)|suggest(?:s|ed|ing)?|"
    r"implies?|which\s+means|we\s+can\s+conclude|based\s+on|given\s+the|"
    r"standard\s+traffic\s+rules?)\b",
    flags=re.I,
)


def has_inference_language(claim: str) -> bool:
    return bool(INFERENCE_LANGUAGE.search(claim))


class EvidenceFirstBlindRead(EvidenceFirstOnlineRepair):
    protocol_version = "evidence_first_v9_candidate_blind_read"

    def _validate_visual_language(self, step):
        if has_inference_language(step.claim):
            raise ValueError(
                "Visual claim contains inference or conclusion language; state only the visible cue"
            )
        return step

    def _visual_proposal(self, image, question, ledger, reason=""):
        return self._validate_visual_language(
            super()._visual_proposal(image, question, ledger, reason)
        )

    def propose(self, image, question, ledger, feedback):
        step = super().propose(image, question, ledger, feedback)
        if step.kind == "visual":
            self._validate_visual_language(step)
        return step

    def _deduction_proposal(self, image, question, ledger):
        prompt = (
            "Using ONLY the labeled ledger, give one concise logical conclusion. "
            "Verified visual facts are premises; uncertain entries are unresolved. "
            "Do not add visual details. Set final_answer only if the premises distinguish "
            "one option; otherwise set it to null and name the missing distinction. "
            "Return JSON with kind=\"deduction\" and evidence_question=\"\".\n"
            f"Question: {question}\nLedger: {self.ledger_text(ledger)}"
        )
        value = parse_json(
            self.vlm.generate(None, prompt, max_tokens=128, schema=V22_DEDUCTION_SCHEMA)
        )
        value["kind"] = "deduction"
        value["evidence_question"] = ""
        from trainingfree.core import Step

        step = Step.from_dict(value)
        self.last_candidate = step
        self.pending_readback = None
        return step

    def audit(self, image, question, step, ledger):
        if step.kind == "deduction":
            return super().audit(image, question, step, ledger)
        if not step.evidence_question.strip():
            return Audit("inconclusive", {}, "Candidate-blind reread needs a neutral inspection question.")
        try:
            box = self.locate(image, step.evidence_question)
            if not box:
                raise ValueError("empty box")
            dx = (box[2] - box[0]) * 0.10
            dy = (box[3] - box[1]) * 0.10
            box = (
                max(0.0, box[0] - dx), max(0.0, box[1] - dy),
                min(1.0, box[2] + dx), min(1.0, box[3] + dy),
            )
            view = [image, self.crop(image, box)]
        except (ValueError, TypeError, KeyError):
            box = ()
            view = image
        prompt = (
            "Inspect the image afresh and answer ONLY the neutral inspection question. "
            "No candidate claim or answer is available. Report one directly visible cue, not "
            "a task option, causal explanation, or guessed conclusion. For counts inspect the "
            "full image for all distinct instances; for orientation report visible front/rear "
            "cues; for relations mention both objects. If the cue is ambiguous, set resolved=false. "
            "Return JSON with observation and resolved.\n"
            f"Inspection question: {step.evidence_question}"
        )
        if box:
            prompt += "\nImage 1 is the full scene; Image 2 is a magnified inspection region."
        blind = parse_json(
            self.vlm.generate(view, prompt, max_tokens=128, schema=BLIND_READ_SCHEMA)
        )
        scores = {"blind_observation": blind, "blind_box": list(box)}
        if not blind.get("resolved"):
            return Audit("inconclusive", scores, "Candidate-blind inspection reports ambiguity.")
        comparison = (
            "Does the independent observation directly and completely support the candidate "
            "visual claim? Reject if the claim adds inference, causal explanation, hidden detail, "
            "or a stronger value than the observation. Reply only Yes or No.\n"
            f"Independent observation: {blind.get('observation', '')}\n"
            f"Candidate visual claim: {step.claim}"
        )
        consistency = self.vlm.yes_probability(None, comparison)
        scores["blind_consistency"] = consistency
        if consistency < self.support:
            return Audit(
                "contradicted" if consistency < 1 - self.support else "inconclusive",
                scores,
                "Candidate-blind observation does not support the proposed visual claim.",
            )
        result = super().audit(image, question, step, ledger)
        result.scores.update(scores)
        return result
