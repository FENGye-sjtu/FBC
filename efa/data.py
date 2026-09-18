"""Acquire real data without treating missing region/step annotations as present."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request


def get_json(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                return json.load(response)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def partition(image_sha):
    bucket = int(image_sha[:8],16) % 100
    return "train" if bucket < 80 else "dev" if bucket < 90 else "calibration"


def image_bytes(value):
    if isinstance(value, list):
        return bytes(value)
    if isinstance(value, str):
        return base64.b64decode(value)
    if isinstance(value, bytes):
        return value
    raise ValueError("Unsupported image representation")


def acquire_deepeyes(output, limit):
    """Small viewer sample for quick integration; freeze by content hashes/revision.

    Full construction must use pinned parquet inputs and cross-source deduplication.
    Viewer responses are checked for truncation, and never assumed to contain bbox.
    """
    from PIL import Image
    repo = "ChenShawn/DeepEyes-Datasets-47k"
    before = get_json(f"https://huggingface.co/api/datasets/{repo}")["sha"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    images = output / "images"
    images.mkdir(exist_ok=True)
    records = []
    preview = None
    for offset in range(0, limit, 4):
        query = urllib.parse.urlencode(dict(dataset=repo, config="default", split="train",
                                            offset=offset, length=min(4,limit-offset)))
        if preview is None:
            # The rows endpoint currently fails for this binary-image parquet;
            # first-rows serves its precomputed preview successfully.
            preview = get_json("https://datasets-server.huggingface.co/first-rows?"+
                urllib.parse.urlencode(dict(dataset=repo, config="default", split="train")))
            if limit > len(preview["rows"]):
                raise ValueError(f"Preview contains only {len(preview['rows'])} rows; use pinned parquet for more")
        payload = {"rows":preview["rows"][offset:min(offset+4,limit)]}
        for item in payload["rows"]:
            if item.get("truncated_cells"):
                print(f"Skipping truncated preview row {item['row_idx']}",flush=True)
                continue
            row = item["row"]
            entries = row["images"]
            if len(entries) != 1:
                continue
            encoded = entries[0]
            if encoded.get("bytes") is not None:
                raw = image_bytes(encoded["bytes"])
            elif encoded.get("src"):
                with urllib.request.urlopen(encoded["src"], timeout=60) as response:
                    raw = response.read()
            else:
                raise ValueError("No downloadable image in viewer row")
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            sha = hashlib.sha256(image.tobytes()+str(image.size).encode()).hexdigest()
            relative = f"images/{sha}.png"
            if not (output/relative).exists():
                image.save(output/relative)
            extra = row.get("extra_info", {})
            question = extra.get("question")
            answer = extra.get("answer",row.get("reward_model",{}).get("ground_truth"))
            if not question or answer is None:
                continue
            records.append(dict(id=f"deepeyes-{item['row_idx']}", source_repo=repo,
                source_revision=before, source_row=item["row_idx"], source_tag=row.get("data_source"),
                image=relative, image_sha256=sha, width=image.width, height=image.height,
                question=question, answer=answer, split=partition(sha),
                annotation_status="raw_qa_requires_step_and_region_annotation",
                benchmark_dedup_status="pending", label_basis="upstream_qa"))
        print(f"Acquired {len(records)} real QA/image records", flush=True)
    after = get_json(f"https://huggingface.co/api/datasets/{repo}")["sha"]
    if before != after:
        raise RuntimeError("Upstream revision changed during viewer acquisition")
    with (output/"raw_tasks.jsonl").open("w",encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    manifest = dict(schema_version="0.1",kind="raw_tasks_not_sft",count=len(records),
                    source_repo=repo,source_revision=before,acquisition="HF viewer, pixel hashes",
                    complete_efa_episodes=0, benchmark_dedup_status="pending")
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


def acquire_metadata(output):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(endpoint=os.environ.get("HF_ENDPOINT","https://hf-mirror.com"))
    sources = {
        "inclusionAI/ZwZ-RL-VQA": ["train.parquet"],
        "deepcs233/Visual-CoT": ["metadata/gqa_cot_train.jsonl"],
    }
    root=Path(output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for repo, names in sources.items():
        info=api.dataset_info(repo)
        for name in names:
            path=hf_hub_download(repo,name,repo_type="dataset",revision=info.sha,
                endpoint=api.endpoint,local_dir=root/repo.replace("/","__"))
            rows.append(dict(repo=repo,revision=info.sha,path=str(Path(path).relative_to(root)),
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), status="metadata_only"))
            print(f"Pinned metadata: {repo}/{name}",flush=True)
    (root/"sources.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["deepeyes","metadata"])
    p.add_argument("--output",required=True)
    p.add_argument("--limit",type=int,default=32)
    a=p.parse_args()
    if a.action=="deepeyes":
        acquire_deepeyes(a.output,a.limit)
    else:
        acquire_metadata(a.output)


if __name__=="__main__":
    main()
