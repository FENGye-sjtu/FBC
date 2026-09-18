"""Shared SFT/runtime serialization for one audited step boundary."""
import json

STEP_END = "<STEP_END>"


def binding_prompt(boxes):
    return ("Identify the object in the FIRST crop relative to its original-frame anchors. "
            "Return its category and spatial/ownership identity. Coordinates: "+json.dumps(boxes))


def premise_prompt(query,premises):
    return "Given ONLY these explicit premises: "+json.dumps(premises)+"\nAnswer: "+json.dumps(query)


def read_prompt(query, boxes):
    return ("Read only the provided source images. Return a short canonical value for the predicate. "
            f"Original-frame coordinates in image order: {boxes}. "
            "Neutral query: "+json.dumps(query))


def route_prompt(query, width, height):
    return ("Locate the original-pixel xyxy evidence regions for this NEUTRAL query. "
            'Return only JSON {"boxes":[[x1,y1,x2,y2]],"context_boxes":[[x1,y1,x2,y2]]}. '
            "Context must include needed anchors, axes, units or relational objects. "
            f"Image size {width}x{height}. Query: " + json.dumps(query))


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
