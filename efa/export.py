"""Export region supervision and reviewed episodes to DeepEyesV2's SFT framework."""
import argparse
import json
from pathlib import Path
from PIL import Image
from efa.format import step_prompt, STEP_END


def export(root, destination, allow_unchecked=False):
    root=Path(root).resolve(); destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"images").mkdir(exist_ok=True)
    raw=[json.loads(line) for line in (root/"raw_tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    outputs={"efa_reads":[],"efa_regions":[],"efa_steps":[]}
    def pair(question,answer,image):
        return {"messages":[{"role":"user","content":"<image>\n"+question},
            {"role":"assistant","content":answer}],"images":[str(image)]}
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
        outputs["efa_reads"].append(pair(row["question"]+"\nReturn a short canonical answer.",row["answer"],path))
        outputs["efa_regions"].append(pair(
            f"Locate the evidence needed for this question. Return original-pixel xyxy coordinates. Image size: {row['width']}x{row['height']}.\n"+row["question"],
            json.dumps({"bbox":row["bbox"]}),root/row["image"]))
    episodes=root/"reviewed_episodes.jsonl"
    if episodes.exists():
        for line in episodes.read_text(encoding="utf-8").splitlines():
            row=json.loads(line)
            if row["split"]!="train" or row.get("annotation_status")!="reviewed_episode":
                continue
            if row.get("benchmark_dedup_status")!="cleared" and not allow_unchecked:
                continue
            context=[]
            for number,step in enumerate(row["steps"],1):
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
