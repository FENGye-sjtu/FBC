"""Freeze explicitly adjudicated episodes and all source/review evidence.

Visual decisions are Codex model annotations, never human gold or measured audits.
This packager validates recorded review evidence; it cannot prove that a visual
judgment is correct or that a view_image tool call occurred. Original review files,
teacher replies and raw records are retained so those claims remain inspectable.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from efa.annotate import validate_steps
from efa.review import usable_sources


LABEL_BASIS = "codex_model_visual_adjudication_not_human_gold_or_runtime_audit"
DECISIONS = {"approve", "revise", "reject", "needs_review"}
FAMILIES = {"attribute", "identity", "text", "relation", "count", "numeric", "deduction"}


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _nonempty(value):
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value) and all(_nonempty(item) for item in value)
    if isinstance(value, dict):
        return bool(value)
    return False


def _inside(root, relative):
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Source artifact escapes root: {relative}")
    return candidate


def _pixel_check(row, root):
    path = _inside(root, row["image"])
    with Image.open(path) as source:
        image = source.convert("RGB")
    if image.size != (row["width"], row["height"]):
        raise ValueError(f"{row['id']}: image dimensions mismatch")
    digest = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
    if digest != row["image_sha256"]:
        raise ValueError(f"{row['id']}: original pixel hash mismatch")
    return path


def validate_episode(episode):
    """Validate exportable structure; this does not certify visual correctness."""
    steps = validate_steps(episode)
    width, height = episode["width"], episode["height"]
    for step in steps:
        parents = step["parents"]
        if any(type(parent) is not int for parent in parents) or len(parents) != len(set(parents)):
            raise ValueError("Parent IDs must be unique integers")
        for atom in step["atoms"]:
            if atom.get("family") not in FAMILIES:
                raise ValueError("Unknown atom family")
            if atom["kind"] == "premise":
                if atom.get("premise_sources", parents) != parents:
                    raise ValueError("Premise sources differ from declared parents")
                atom["premise_sources"] = list(parents)
                continue
            region = atom.get("region", {})
            if region.get("insufficient") is not False:
                raise ValueError("Accepted observation has insufficient evidence")
            if region.get("coordinate_system") not in {
                "original_pixels", "original_pixels_converted_from_normalized_1000"
            }:
                raise ValueError("Region must use original-image pixel coordinates")
            for field in ("boxes", "context_boxes"):
                boxes = region.get(field)
                if not isinstance(boxes, list) or (field == "boxes" and not boxes):
                    raise ValueError(f"Invalid {field}")
                for box in boxes:
                    if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
                        raise ValueError("Boxes must contain four integer coordinates")
                    x1, y1, x2, y2 = box
                    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                        raise ValueError("Box outside original image or empty")
            if not region["context_boxes"]:
                # Runtime routing requires a nonempty context. Reusing the
                # already inspected target adds no new pixels or visual claim.
                region["context_boxes"] = copy.deepcopy(region["boxes"])
                region["context_serialization_note"] = "Empty optional context normalized to the inspected target boxes for runtime routing."
    if not usable_sources(dict(episode, steps=steps)):
        raise ValueError("Episode has unusable visual queries")
    answer = episode.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Accepted episode requires a canonical string answer")
    if steps[-1]["atoms"][-1]["value"].strip() != answer.strip():
        raise ValueError("Final atom value does not match adjudicated answer")
    return steps


def _review_evidence(raw, review):
    checks = review.get("source_checks") or {}
    view_evidence = review.get("views_inspected") or review.get("visual_review_evidence") or checks.get("visual_review_evidence")
    for field in ("visual_findings", "step_count_reason"):
        if not _nonempty(review.get(field)):
            raise ValueError(f"{raw['id']}: missing {field}")
    if not _nonempty(view_evidence):
        raise ValueError(f"{raw['id']}: missing recorded visual inspection evidence")


def _episode(raw, review, review_file):
    checks = review.get("source_checks") or {}
    question = review.get("question", raw["question"])
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Adjudicated question must be nonempty")
    changed = question != raw["question"]
    if review.get("question_revised", False) is not changed:
        raise ValueError(f"{raw['id']}: question change must be explicitly marked")
    reason = review.get("question_revision_reason") or checks.get("question_revision_reason")
    if changed and not _nonempty(reason):
        raise ValueError(f"{raw['id']}: question rewrite needs a reason")
    row = copy.deepcopy(raw)
    row.update(
        question=question, answer=review.get("answer"), steps=copy.deepcopy(review.get("steps")),
        source_question=raw["question"], source_answer=raw["answer"], source_split=raw["split"],
        source_annotation_status=raw.get("annotation_status"), source_label_basis=raw.get("label_basis"),
        question_revised=changed, answer_revised=review.get("answer") != raw["answer"],
        annotation_status="reviewed_episode", label_basis=LABEL_BASIS,
        codex_review=copy.deepcopy(review), codex_review_file=review_file,
        runtime_audit=None,
        engineering_only=raw.get("benchmark_dedup_status") != "cleared",
    )
    if changed:
        row["question_revision"] = dict(original=raw["question"], revised=question, reason=reason)
    row["steps"] = validate_episode(row)
    return row


def package(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite data version: {output}")
    if root == output or root.is_relative_to(output):
        raise ValueError("Output must not contain the input root")
    raw_path, draft_path = root / "raw_tasks.jsonl", root / "teacher_draft_results.jsonl"
    raw = _records(raw_path)
    sources = {}
    images = {}
    for row in raw:
        if row["id"] in sources:
            raise ValueError(f"Duplicate raw task: {row['id']}")
        if row.get("split") not in {"train", "dev", "calibration", "test", "validation"}:
            raise ValueError(f"Unknown source split: {row.get('split')}")
        sources[row["id"]] = row
        images[row["image"]] = _pixel_check(row, root)
    drafts = _records(draft_path)
    draft_ids = set()
    for draft in drafts:
        if draft["id"] not in sources or draft["id"] in draft_ids:
            raise ValueError("Unknown or duplicate teacher draft ID")
        draft_ids.add(draft["id"])
    reviews, review_files, reviewed_ids, accepted = [], [], set(), []
    for path in sorted((root / "codex_reviews").glob("*.json")):
        entries = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError(f"Review file must contain a record list: {path.name}")
        relative = path.relative_to(root).as_posix()
        review_files.append((relative, path))
        for review in entries:
            rid = review["id"]
            if rid not in sources or rid in reviewed_ids:
                raise ValueError(f"Unknown or duplicate review ID: {rid}")
            raw_row = sources[rid]
            if review.get("decision") not in DECISIONS:
                raise ValueError(f"{rid}: review decision must be explicit")
            if not str(review.get("reviewer", "")).startswith("codex"):
                raise ValueError(f"{rid}: review must identify the Codex model annotator")
            if review.get("image_sha256") != raw_row["image_sha256"]:
                raise ValueError(f"{rid}: review image hash mismatch")
            if review.get("split", raw_row["split"]) != raw_row["split"]:
                raise ValueError(f"{rid}: review cannot change original split")
            if review.get("runtime_audit") is not None:
                raise ValueError(f"{rid}: offline review cannot supply runtime audit results")
            _review_evidence(raw_row, review)
            reviewed_ids.add(rid)
            reviews.append(dict(review, review_source_file=relative))
            if review["decision"] in {"approve", "revise"}:
                accepted.append(_episode(raw_row, review, relative))
    decisions = Counter(review["decision"] for review in reviews)
    cache_files = [(path.relative_to(root).as_posix(), path) for path in sorted((root / "teacher_cache").rglob("*")) if path.is_file()]
    artifacts = [("raw_tasks.jsonl", raw_path), ("teacher_draft_results.jsonl", draft_path)]
    inspection_files = [(path.relative_to(root).as_posix(), path)
                        for path in sorted((root / "inspection").rglob("*")) if path.is_file()]
    artifacts += sorted(images.items()) + review_files + cache_files + inspection_files
    # Capture input bytes before mutation. Any validation error leaves no new version.
    artifact_bytes = {}
    for relative, path in artifacts:
        checked = _inside(root, relative)
        if checked != path.resolve():
            raise ValueError("Source path mismatch")
        artifact_bytes[relative] = checked.read_bytes()
    manifest = dict(
        schema_version="0.1", kind="codex_visually_adjudicated_episodes",
        count=len(accepted), raw_tasks=len(raw), reviews=len(reviews),
        complete_efa_episodes=len(accepted), complete_steps=sum(len(row["steps"]) for row in accepted),
        step_distribution=dict(sorted(Counter(str(len(row["steps"])) for row in accepted).items())),
        decisions=dict(sorted(decisions.items())), unreviewed_tasks=len(sources) - len(reviewed_ids),
        accepted_split_counts=dict(sorted(Counter(row["split"] for row in accepted).items())),
        teacher_draft_records=len(drafts), teacher_draft_successes=sum("draft" in row for row in drafts),
        teacher_draft_errors=sum("error" in row for row in drafts), teacher_cache_files=len(cache_files),
        images=len(images), image_edits=False, label_basis=LABEL_BASIS,
        source_answer_disagreements=sum(row["answer_revised"] for row in accepted),
        question_rewrites=sum(row["question_revised"] for row in accepted),
        engineering_only=any(row.get("benchmark_dedup_status") != "cleared" for row in raw),
        benchmark_dedup_status="cleared" if raw and all(row.get("benchmark_dedup_status") == "cleared" for row in raw) else "pending",
        runtime_audited_episodes=0, human_gold_episodes=0, construction_complete=True,
        visual_review_evidence_policy="Recorded view evidence required for acceptance; record presence does not programmatically prove tool use or visual correctness.",
        source_artifact_sha256={name: hashlib.sha256(content).hexdigest() for name, content in artifact_bytes.items()},
    )
    output.mkdir(parents=True, exist_ok=False)
    for name, content in artifact_bytes.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    # Pixel hashes were validated against source files. Copies must be byte-identical.
    for name, expected in manifest["source_artifact_sha256"].items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Copied artifact differs: {name}")
    for row in raw:
        _pixel_check(row, output)
    for name, rows in (("reviewed_episodes.jsonl", accepted), ("adjudication_records.jsonl", reviews)):
        (output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = package(args.root, args.output)
    print(json.dumps({k: v for k, v in manifest.items() if k != "source_artifact_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
