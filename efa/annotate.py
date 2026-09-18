"""Offline teacher annotations with a separate value-blind region request.

Teacher outputs are pseudo-labels until independently checked. This module does
not invent audit scores, PASS labels, or repair outcomes from annotation truth.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request
from efa.regions import Box, complement

STEP_PROMPT = '''Write a concise image-grounded solution as numbered steps.
Return JSON only: {"steps":[{"id":1,"text":"Step 1: ...", "parents":[],
"atoms":[{"subject":"stable neutral referent","predicate":"color",
"value":"blue","family":"attribute","kind":"visual"}]}]}.
Use families attribute, identity, text, relation, count, numeric, deduction.
Each step has 1-3 atoms. Use 2-8 steps only where the task needs them; do not pad
simple observations with invented facts. Parents are earlier step IDs.
For deductions, specify the actual exposed premises. Every observation must be
checkable in the image; no answer-conditioned invented observations.
The subject MUST NOT include the disputed value. For a color claim, use stable
position/ownership/category, never "the red bag". Final step answers the question.
No audit scores or verification decisions. Reference answers are annotation aids,
not proof of intermediate claims. Flag uncertainty rather than inventing evidence.'''

REGION_PROMPT = '''Locate evidence for the neutral subject/predicate queries.
You have not been given their asserted values. Return JSON only:
{"regions":[{"key":"1.0","boxes":[[x1,y1,x2,y2]],
"context_boxes":[[x1,y1,x2,y2]],"identity":"neutral marked-target identity",
"insufficient":false}]}.
Coordinates are integer ORIGINAL-image pixels, xyxy. Include necessary relation
anchors, scale, units, legend, or counting extent. Multiple regions or whole image
are allowed when necessary. A crop existing is not evidence of claim correctness.
Return insufficient=true if the referent cannot be identified. Never infer an
answer merely from the question. Include every query key exactly once.'''


def teacher_json(messages, cache, stage):
    endpoint = os.environ.get("EFA_TEACHER_BASE_URL", "").rstrip("/")
    model = os.environ.get("EFA_TEACHER_MODEL", "")
    key = os.environ.get("EFA_TEACHER_API_KEY", "")
    if not endpoint or not model or not key:
        raise RuntimeError("Set EFA_TEACHER_BASE_URL, EFA_TEACHER_MODEL and EFA_TEACHER_API_KEY")
    if urllib.parse.urlparse(endpoint).scheme != "https":
        raise ValueError("Teacher credentials require an HTTPS endpoint")
    body = dict(model=model, messages=messages, temperature=0, max_tokens=4096,
                response_format={"type":"json_object"})
    encoded = json.dumps(body).encode()
    digest = hashlib.sha256(endpoint.encode()+encoded).hexdigest()
    path=cache/(digest+".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    request=urllib.request.Request(endpoint+"/chat/completions",data=encoded,
        headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
    with urllib.request.urlopen(request,timeout=180) as response:
        result=json.load(response)
    content=result["choices"][0]["message"]["content"]
    parsed=json.loads(content)
    record=dict(output=parsed, model=model, endpoint_host=urllib.parse.urlparse(endpoint).hostname,
                usage=result.get("usage"),request_sha256=digest,stage=stage,created_at=time.time())
    path.write_text(json.dumps(record,ensure_ascii=False),encoding="utf-8")
    return record


def validate_steps(output):
    steps=output.get("steps",[])
    if not 1 <= len(steps) <= 16:
        raise ValueError("Invalid step count")
    seen=set()
    for step in steps:
        sid=step["id"]
        if sid in seen or not isinstance(sid,int) or not step["text"].startswith(f"Step {sid}:"):
            raise ValueError("Invalid numbered step")
        if any(parent not in seen for parent in step["parents"]):
            raise ValueError("Dependency is not an earlier step")
        atoms=step["atoms"]
        if not 1 <= len(atoms) <= 3:
            raise ValueError("Each step must have 1-3 atoms")
        for atom in atoms:
            if not all(isinstance(atom.get(k),str) and atom[k].strip() for k in ("subject","predicate","value")):
                raise ValueError("Empty or malformed atom")
            if atom.get("kind") not in ("visual","premise"):
                raise ValueError("Unknown evidence mode")
            if atom["kind"]=="premise" and not step["parents"]:
                raise ValueError("Premise claim has no parents")
        seen.add(sid)
    return steps


def annotate_task(task, root, cache):
    image_path=root/task["image"]
    data_url="data:image/png;base64,"+base64.b64encode(image_path.read_bytes()).decode()
    def request(prompt,text):
        return [{"role":"system","content":prompt},{"role":"user","content":[
            {"type":"image_url","image_url":{"url":data_url}}, {"type":"text","text":text}]}]
    first=teacher_json(request(STEP_PROMPT,json.dumps({"question":task["question"],
        "reference_answer":task["answer"]})),cache,"step_parse")
    steps=validate_steps(first["output"])
    queries=[dict(key=f"{s['id']}.{i}",subject=a["subject"],predicate=a["predicate"])
             for s in steps for i,a in enumerate(s["atoms"]) if a["kind"]=="visual"]
    # New conversation: no step text, incumbent values, answers, or first response.
    second=teacher_json(request(REGION_PROMPT,json.dumps({"width":task["width"],
        "height":task["height"],"queries":queries})),cache,"value_blind_route")
    entries=second["output"]["regions"]
    regions={r["key"]:r for r in entries}
    if len(regions)!=len(entries) or set(regions)!={q["key"] for q in queries}:
        raise ValueError("Region coverage is not exact")
    for step in steps:
        for i,atom in enumerate(step["atoms"]):
            if atom["kind"]=="premise":
                # Deduction is checked against explicit statements, whose visual
                # provenance is inherited, not a newly invented arithmetic box.
                atom["premise_sources"]=step["parents"]
                continue
            region=regions[f"{step['id']}.{i}"]
            boxes=[Box(*b) for b in region["boxes"]]
            contexts=[Box(*b) for b in region["context_boxes"]]
            if not boxes or not contexts or not all(b.valid(task["width"],task["height"]) for b in boxes+contexts):
                raise ValueError("Invalid or missing evidence region")
            atom.update(region=region,complement_boxes=[list(vars(b).values()) for b in
                complement(task["width"],task["height"],boxes)])
    return dict(task,steps=steps,annotation_status="teacher_pseudo_episode_requires_review",
                provenance=[{k:v for k,v in x.items() if k!="output"} for x in (first,second)],
                runtime_audit=None)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--limit",type=int,default=32)
    a=p.parse_args()
    root=Path(a.root); cache=root/"teacher_cache"; cache.mkdir(exist_ok=True)
    tasks=[json.loads(s) for s in (root/"raw_tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    with (root/"pseudo_episodes.jsonl").open("w",encoding="utf-8") as out:
        for task in tasks[:a.limit]:
            result=annotate_task(task,root,cache)
            out.write(json.dumps(result,ensure_ascii=False)+"\n"); out.flush()
            print(f"Annotated {task['id']}: {len(result['steps'])} steps",flush=True)


if __name__=="__main__":
    main()
