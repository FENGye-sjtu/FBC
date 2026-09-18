"""Offline teacher annotations with a separate value-blind region request.

Teacher outputs are pseudo-labels until independently checked. This module does
not invent audit scores, PASS labels, or repair outcomes from annotation truth.
"""
import argparse
import base64
import copy
import hashlib
import json
import mimetypes
import math
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request
from efa.regions import Box, complement, normalized_boxes

STEP_PROMPT = '''Write a concise image-grounded solution as numbered steps.
Return JSON only: {"steps":[{"id":1,"text":"Step 1: ...", "parents":[],
"atoms":[{"subject":"stable neutral referent","predicate":"color",
"value":"blue","family":"attribute","kind":"visual"}]}]}.
Use families attribute, identity, text, relation, count, numeric, deduction.
Each step has 1-3 atoms. Use one step for direct attribute questions. Use multiple
steps only for genuine dependencies such as comparison, counting or calculation.
Do not invent identity or appearance facts to pad a simple answer. Do not repeat
the final visual answer in an unnecessary deduction step. Parents are earlier step IDs.
kind MUST be exactly "visual" or "premise"; use kind="premise" for deductions.
For deductions, specify the actual exposed premises. Every observation must be
checkable in the image; no answer-conditioned invented observations.
The subject MUST NOT include the disputed value. For a color claim, use stable
position/ownership/category, never "the red bag". Final step answers the question.
The final atom value must be the short canonical answer, without a sentence wrapper.
When a source program is supplied, use its actual visual dependencies: first
establish necessary referents/relations, then read the queried property. Each
step must assert a checkable fact, not an instruction to look for something.
Every neutral subject/predicate query must stand alone without the step text.
For relations, include BOTH entities: e.g. subject="person on bicycle",
predicate="horizontal position relative to pedestrians near storefront",
value="left". Never use the bare predicate "relation" or leave the anchor implicit.
No audit scores or verification decisions. Reference answers are annotation aids,
not proof of intermediate claims. Flag uncertainty rather than inventing evidence.'''

REGION_PROMPT = '''Locate evidence for the neutral subject/predicate queries.
You have not been given their asserted values. Return JSON only:
{"regions":[{"key":"1.0","boxes":[[x1,y1,x2,y2]],
"context_boxes":[[x1,y1,x2,y2]],"identity":"neutral marked-target identity",
"insufficient":false}]}.
Coordinates are integers NORMALIZED to 0..1000, xyxy: x/width*1000 and y/height*1000.
Use the supplied native candidate boxes when they refer to the queried object;
they provide locations, not answers or proof of support. Include necessary relation
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
    if model.startswith('qwen3'):
        body['enable_thinking']=False
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
    # Normalize serialization only; never rewrite an observation or dependency.
    # Keep the original teacher response immutable in the request cache.
    steps=copy.deepcopy(output.get("steps",[]))
    if not 1 <= len(steps) <= 16:
        raise ValueError("Invalid step count")
    seen=set()
    for step in steps:
        sid=step["id"]
        if type(sid) is not int or sid!=len(seen)+1 or not isinstance(step['text'],str) or not step['text'].strip():
            raise ValueError("Invalid numbered step")
        if not step['text'].startswith(f'Step {sid}:'):
            if step['text'].startswith('Step '):raise ValueError('Conflicting step numbering')
            step['text']=f'Step {sid}: '+step['text']
        if any(parent not in seen for parent in step["parents"]):
            raise ValueError("Dependency is not an earlier step")
        atoms=step["atoms"]
        if not 1 <= len(atoms) <= 3:
            raise ValueError("Each step must have 1-3 atoms")
        for atom in atoms:
            value=atom.get('value')
            if type(value) in (int,float) and math.isfinite(value):
                atom['value']=str(value)
            if atom.get('kind')=='deduction':
                atom['kind']='premise'
            if not all(isinstance(atom.get(k),str) and atom[k].strip() for k in ("subject","predicate","value")):
                raise ValueError("Empty or malformed atom")
            if atom.get("kind") not in ("visual","premise"):
                raise ValueError("Unknown evidence mode")
            if atom['predicate'].strip().lower()=='relation':
                raise ValueError('Relation query must specify its reference entity and relation type')
            if atom["kind"]=="premise" and not step["parents"]:
                raise ValueError("Premise claim has no parents")
        seen.add(sid)
    return steps


def annotate_task(task, root, cache):
    image_path=root/task["image"]
    mime=mimetypes.guess_type(image_path.name)[0] or 'image/png'
    data_url="data:"+mime+";base64,"+base64.b64encode(image_path.read_bytes()).decode()
    def request(prompt,text):
        return [{"role":"system","content":prompt},{"role":"user","content":[
            {"type":"image_url","image_url":{"url":data_url}}, {"type":"text","text":text}]}]
    first=teacher_json(request(STEP_PROMPT,json.dumps({"question":task["question"],
        "reference_answer":task["answer"],"source_program":task.get('source_program')})),cache,"step_parse")
    steps=validate_steps(first["output"])
    queries=[dict(key=f"{s['id']}.{i}",subject=a["subject"],predicate=a["predicate"])
             for s in steps for i,a in enumerate(s["atoms"]) if a["kind"]=="visual"]
    # New conversation: no step text, incumbent values, answers, or first response.
    second=teacher_json(request(REGION_PROMPT,json.dumps({"width":task["width"],
        "height":task["height"],"queries":queries,
        "native_candidate_boxes_1000":[[round(v/(task['width'] if i%2==0 else task['height'])*1000)
            for i,v in enumerate(box)] for box in (task.get('native_candidate_boxes') or ([task['bbox']] if task.get('bbox') else []))]})),cache,"value_blind_route_normalized_1000")
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
            for field in ['boxes','context_boxes']:
                region[field]=normalized_boxes(region[field],task['width'],task['height'])
            region['coordinate_system']='original_pixels_converted_from_normalized_1000'
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
    destination=root/'pseudo_episodes.jsonl'
    existing={json.loads(s)['id'] for s in destination.read_text(encoding='utf-8').splitlines()} if destination.exists() else set()
    with destination.open("a",encoding="utf-8") as out:
        for task in tasks[:a.limit]:
            if task['id'] in existing:
                continue
            try:
                result=annotate_task(task,root,cache)
            except (ValueError,KeyError,TypeError) as error:
                with (root/'annotation_rejections.jsonl').open('a',encoding='utf-8') as rejected:
                    rejected.write(json.dumps(dict(id=task['id'],reason=str(error)))+'\n')
                print(f"Rejected {task['id']}: {error}",flush=True)
                continue
            out.write(json.dumps(result,ensure_ascii=False)+"\n"); out.flush()
            print(f"Annotated {task['id']}: {len(result['steps'])} steps",flush=True)


if __name__=="__main__":
    main()
