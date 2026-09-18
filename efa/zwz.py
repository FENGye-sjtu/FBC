"""Convert ZwZ into original-pixel region training records (not invented CoT)."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tarfile
import urllib.request
from PIL import Image
import pyarrow.parquet as pq
from efa.data import partition
from efa.regions import Box, complement


def canonical_qa(row):
    text=row["extra_info"]["question"]
    options=dict(re.findall(r"(?m)^([A-Z])\.\s*(.+)$",text))
    letter=row["answer"].strip()
    if letter not in options:
        raise ValueError("Unmatched answer choice")
    value=options[letter].strip().strip("* ")
    question=re.split(r"(?m)^[A-Z]\.\s",text)[0]
    question=re.sub(r"Only focus on the objects inside the red bounding box in the image to answer this question\.","",question)
    question=question.replace("<image>","").strip()
    if not value or not question:
        raise ValueError("Empty canonical value/query")
    return question,value


class Shards:
    """Read split gzip parts in order without downloading the entire archive."""
    def __init__(self, urls):
        self.urls=iter(urls)
        self.current=None

    def read(self,n=-1):
        while True:
            if self.current is None:
                try:
                    url=next(self.urls)
                except StopIteration:
                    return b""
                self.current=urllib.request.urlopen(url,timeout=120)
            data=self.current.read(n)
            if data:
                return data
            self.current.close(); self.current=None

    def close(self):
        if self.current:
            self.current.close()


def build(metadata, output, limit, endpoint):
    from huggingface_hub import HfApi
    root=Path(output).resolve(); root.mkdir(parents=True,exist_ok=True)
    (root/"images").mkdir(exist_ok=True)
    rows=pq.read_table(metadata).to_pylist()
    grouped={}
    for index,row in enumerate(rows):
        try:
            q,v=canonical_qa(row)
        except ValueError:
            continue
        name=Path(row["original_images"][0]).name
        grouped.setdefault(name,[]).append((index,row,q,v))
    api=HfApi(endpoint=endpoint)
    info=api.dataset_info("inclusionAI/ZwZ-RL-VQA")
    files=sorted(s.rfilename for s in info.siblings if s.rfilename.startswith("original_images/original_images.tar.gz"))
    if not files:
        raise RuntimeError("Original-image archive parts unavailable")
    urls=[f"{endpoint}/datasets/inclusionAI/ZwZ-RL-VQA/resolve/{info.sha}/{name}" for name in files]
    source=Shards(urls)
    records=[]; visited=set()
    try:
        # Upstream files are named .tar.gz00 but their magic is uncompressed tar.
        # Detect compression from bytes, not the misleading suffix.
        with tarfile.open(fileobj=source,mode="r|*") as archive:
            for member in archive:
                name=Path(member.name).name
                if not member.isfile() or name not in grouped or name in visited:
                    continue
                visited.add(name)
                raw=archive.extractfile(member).read()
                image=Image.open(io.BytesIO(raw)).convert("RGB")
                sha=hashlib.sha256(image.tobytes()+str(image.size).encode()).hexdigest()
                path=f"images/{sha}.png"
                image.save(root/path)
                for index,row,question,value in grouped[name]:
                    x1,y1,x2,y2=row["bbox"]
                    box=Box(math.floor(x1),math.floor(y1),math.ceil(x2),math.ceil(y2))
                    if not box.valid(image.width,image.height):
                        continue
                    # No guessed multi-step rationale: this is a region QA atom.
                    record=dict(id=f"zwz-{index}",source_repo="inclusionAI/ZwZ-RL-VQA",
                        source_revision=info.sha,source_image=name,image=path,image_sha256=sha,
                        width=image.width,height=image.height,question=question,answer=value,
                        original_choice=row["answer"],bbox=list(vars(box).values()),
                        context_bbox=list(vars(box.expand(image.width,image.height)).values()),
                        complement_boxes=[list(vars(b).values()) for b in complement(image.width,image.height,[box])],
                        split=partition(sha),label_basis="ZwZ_teacher_consensus",
                        annotation_status="region_qa_seed",benchmark_dedup_status="pending",
                        steps=None,runtime_audit=None)
                    records.append(record)
                    if len(records)>=limit:
                        break
                print(f"ZwZ: {len(records)} region QA records from {len(visited)} original images",flush=True)
                if len(records)>=limit:
                    break
    finally:
        source.close()
    if not records:
        raise RuntimeError("No matched original images found")
    with (root/"raw_tasks.jsonl").open("w",encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    manifest=dict(kind="region_qa_seed",source_revision=info.sha,count=len(records),images=len(visited),
                  complete_efa_episodes=0,benchmark_dedup_status="pending",image_edits=False)
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--metadata",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--limit",type=int,default=64)
    p.add_argument("--endpoint",default=os.environ.get("HF_ENDPOINT","https://hf-mirror.com"))
    a=p.parse_args()
    build(a.metadata,a.output,a.limit,a.endpoint)
