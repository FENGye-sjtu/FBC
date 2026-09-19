"""Export region supervision and reviewed episodes to DeepEyesV2's SFT framework."""
import argparse
import json
from pathlib import Path
from PIL import Image
from efa.format import step_prompt, route_prompt, read_prompt, binding_prompt, premise_prompt, STEP_END
from efa.native import property_atom
from efa.review import usable_sources


def export(root, destination, allow_unchecked=False):
    root=Path(root).resolve(); destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"images").mkdir(exist_ok=True)
    raw=[json.loads(line) for line in (root/"raw_tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    outputs={"efa_reads":[],"efa_regions":[],"efa_atoms":[],"efa_steps":[],"efa_premises":[],"efa_bindings":[]}
    def pair(question,answer,image):
        images=image if isinstance(image,list) else [image]
        return {"messages":[{"role":"user","content":"<image>\n"*len(images)+question},
            {"role":"assistant","content":answer}],"images":[str(im) for im in images]}
    for row in raw:
        if row["split"]!="train":
            continue
        if row.get("benchmark_dedup_status")!="cleared" and not allow_unchecked:
            continue
        if not row.get("bbox"):
            continue
        crop=Image.open(root/row["image"]).crop(tuple(row["bbox"]))
        path=destination/"images"/(row["id"]+".png")
        crop.save(path)
        atom=property_atom(row['question'],row['answer'])
        prompt=read_prompt({k:atom[k] for k in ('subject','predicate')},[row['bbox']]) if atom else row['question']+'\nReturn a short canonical answer.'
        outputs["efa_reads"].append(pair(prompt,row["answer"],path))
        if atom:
            # Native object boxes supervise the target only. Full-frame context
            # preserves anchors without inventing a tighter annotated context.
            query={k:atom[k] for k in ('subject','predicate')}
            outputs['efa_regions'].append(pair(
                route_prompt(query,row['width'],row['height']),
                json.dumps(dict(boxes=[row['bbox']],
                    context_boxes=[row['bbox'],[0,0,row['width'],row['height']]])),root/row['image']))
            # Single-observation format supervision, not an invented multi-step
            # rationale or a reviewed runtime audit episode.
            target=json.dumps(dict(atoms=[atom],final=True))+STEP_END
            outputs['efa_atoms'].append(pair(step_prompt(row['question'],[],1),target,root/row['image']))
    episode_files=[root/'reviewed_episodes.jsonl']
    if allow_unchecked:
        episode_files.append(root/'teacher_reviewed_episodes.jsonl')
        episode_files.append(root/'native_episodes.jsonl')
    for episodes in episode_files:
        if not episodes.exists():continue
        for line in episodes.read_text(encoding="utf-8").splitlines():
            row=json.loads(line)
            allowed=['reviewed_episode']+(['teacher_reviewed_episode','native_program_episode'] if allow_unchecked else [])
            if row["split"]!="train" or row.get("annotation_status") not in allowed:
                continue
            if not usable_sources(row):
                continue
            if row.get("benchmark_dedup_status")!="cleared" and not allow_unchecked:
                continue
            context=[]
            for number,step in enumerate(row["steps"],1):
                for index,atom in enumerate(step['atoms']):
                    if atom['kind']=='premise':
                        query={k:atom[k] for k in ('subject','predicate')}
                        outputs['efa_premises'].append(pair(premise_prompt(query,context),atom['value'],[]))
                    if atom['kind']!='visual':continue
                    query={k:atom[k] for k in ('subject','predicate')}
                    region=atom['region']
                    outputs['efa_regions'].append(pair(route_prompt(query,row['width'],row['height']),
                        json.dumps({k:region[k] for k in ('boxes','context_boxes')}),root/row['image']))
                    paths=[]
                    with Image.open(root/row['image']) as original:
                        for view,box in enumerate(region['boxes']):
                            path=destination/'images'/f"{row['id']}-s{number}-a{index}-v{view}.png"
                            original.crop(tuple(box)).save(path);paths.append(path)
                    outputs['efa_reads'].append(pair(read_prompt(query,region['boxes']),atom['value'],paths))
                    # Native object provenance identifies the first crop's
                    # subject. Whole-scene count sources do not provide this.
                    if atom.get('native_evidence',{}).get('object_ids') and row.get('native_program_objects'):
                        boxes=[list(b) for b in dict.fromkeys(tuple(b) for b in region['boxes']+region['context_boxes'])]
                        binding_paths=[]
                        with Image.open(root/row['image']) as original:
                            for view,box in enumerate(boxes):
                                path=destination/'images'/f"{row['id']}-s{number}-a{index}-binding-v{view}.png"
                                original.crop(tuple(box)).save(path);binding_paths.append(path)
                        outputs['efa_bindings'].append(pair(binding_prompt(boxes),atom['subject'],binding_paths))
                atoms=[{k:a[k] for k in ("subject","predicate","value","family","kind")} for a in step["atoms"]]
                target=json.dumps(dict(atoms=atoms,final=number==len(row["steps"])))+STEP_END
                outputs["efa_steps"].append(pair(step_prompt(row["question"],context,number),target,root/row["image"]))
                parents=[p["key"] for p in context]
                context.extend(dict(key=f"step{number}.atom{i}@1",claim=dict(a,parents=parents),verified=None)
                               for i,a in enumerate(atoms))
    info={}
    for name,rows in outputs.items():
        (destination/(name+".json")).write_text(json.dumps(rows,ensure_ascii=False),encoding="utf-8")
        info[name]={"file_name":name+".json","formatting":"sharegpt",
                    "columns":{"messages":"messages","images":"images"},
                    "tags":{"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant"}}
    (destination/"dataset_info.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    manifest={"counts":{k:len(v) for k,v in outputs.items()},"engineering_only":allow_unchecked,
              "benchmark_dedup_override":allow_unchecked}
    (destination/"export_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--output",required=True)
    p.add_argument("--engineering-only",action="store_true")
    a=p.parse_args();export(a.root,a.output,a.engineering_only)
