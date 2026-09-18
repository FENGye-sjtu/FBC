"""Shared SFT/runtime serialization for one audited step boundary."""
import json

STEP_END = "<STEP_END>"


def step_prompt(question, context, number, final_budget=False):
    prompt=(f"Question: {question}\nAccepted/provisional ledger: {json.dumps(context)}\n"
        f"Generate exactly Step {number}, then stop. Return JSON followed by {STEP_END}: "
        '{"atoms":[{"subject":"neutral referent","predicate":"property","value":"answer",'
        '"family":"attribute","kind":"visual"}],"final":false}. '
        "Maximum three atoms. For inference use kind=premise and family=deduction. "
        "When ready, final=true and the final answer must be the value of the LAST atom. "
        "Do not include disputed values in object identities. Earlier unverified values are provisional. "
        'If later evidence requires rechecking an earlier premise, instead return {"review_parent":"exact live node key"} '
        f'followed by {STEP_END}. Do not propose its replacement before the audit. ')
    if final_budget:
        prompt+="This is the final budgeted step: supply the final answer."
    return prompt
