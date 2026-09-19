"""Freeze reviewed attribute/category contrasts separately from repair trajectories.

This packages existing Codex visual judgments and exact, separately approved
runtime prompts. It does not run a verifier, invent student repairs or modify
positive episodes. Coordinate exposure is approved only for the listed pairs;
the two pixels-only insufficient candidates stay outside training.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from efa.data import partition
from efa.format import binding_prompt, read_prompt


LABEL_BASIS = "codex_model_visual_adjudication_not_human_gold_or_runtime_audit"
APPROVED = "approved_semantics_unchanged_with_runtime_coordinates"
COLORS = {"red", "blue", "yellow", "black", "white", "green", "gray", "grey",
          "orange", "purple", "pink", "brown", "tan", "beige"}


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _jsonl_bytes(rows):
    return b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in rows)


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing source artifact or path outside root: {relative}")
    return path


def _lines(path):
    result = {}
    for line_number, raw in enumerate(path.read_bytes().splitlines(keepends=True), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if row["id"] in result:
            raise ValueError(f"Duplicate source ID: {row['id']}")
        result[row["id"]] = (row, raw, line_number)
    return result


def _atom(episode, step_id, index):
    steps = [step for step in episode["steps"] if step["id"] == step_id]
    if len(steps) != 1 or type(index) is not int or not 0 <= index < len(steps[0]["atoms"]):
        raise ValueError("Invalid positive step/atom reference")
    return steps[0]["atoms"][index]


def _boxes(boxes, width, height):
    if not isinstance(boxes, list) or not boxes:
        raise ValueError("Missing source boxes")
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError("Boxes must contain four integer coordinates")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("Source box is empty or outside the original image")


def _no_value_leak(prompt, values):
    text = " ".join(prompt.casefold().split())
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Pair targets must be nonempty strings")
        value = " ".join(value.casefold().split())
        if re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text):
            raise ValueError("A supervised candidate value leaked into the runtime prompt")


def build_pair(record, variant, episode, approval):
    """Return a validated training contrast, or None for an excluded candidate."""
    if variant["type"] == "missing_object_anchor":
        if variant["training_eligibility"] != "insufficient_candidate_only":
            raise ValueError("Insufficient evidence cannot be promoted to a training pair")
        return None
    if record["split"] != "train" or episode["split"] != "train":
        raise ValueError("Training pair contains a held-out image group")
    if record["image_sha256"] != episode["image_sha256"]:
        raise ValueError("Pair and positive episode image hashes differ")
    atom = _atom(episode, variant["positive_step_id"], variant["positive_atom_index"])
    boxes = record["sources"][variant["source_id"]]["boxes"]
    labels = variant["labels"]
    if variant["type"] == "correct_source_wrong_attribute":
        if (variant["training_eligibility"] != "attribute_ce_ul"
                or labels.get("attribute") != "contradicted"
                or labels.get("binding") != "not_supervised"
                or variant["corrected_value"] != atom["value"]
                or boxes != atom["region"]["boxes"]):
            raise ValueError("Attribute contrast differs from its reviewed positive source")
        kind, family = "attribute", "runtime_read"
        query = {key: atom[key] for key in ("subject", "predicate")}
        positive, negative = variant["corrected_value"], variant["incumbent_value"]
        prompt = read_prompt(query, boxes)
    elif variant["type"] == "wrong_object_same_attribute":
        other = _atom(episode, variant["source_from_positive_step_id"], 0)
        if (variant["training_eligibility"] != "binding_ce_ul_only"
                or labels.get("binding") != "contradicted"
                or labels.get("local_attribute") != "supported"
                or variant["incumbent_value"] != atom["value"]
                or atom["value"] != other["value"]):
            raise ValueError("Binding-only contrast must preserve the local attribute")
        kind, family, query = "binding", "runtime_binding", None
        positive = variant["actual_object_identity"]
        negative = variant["impersonated_object_identity"]
        if any(value.casefold().strip() in COLORS for value in (positive, negative)):
            raise ValueError("Binding contrast targets must be object categories, never colors")
        prompt = binding_prompt(boxes)
    else:
        raise ValueError("Unknown correction candidate kind")
    if positive == negative:
        raise ValueError("Positive and negative targets must differ")
    _no_value_leak(prompt, (positive, negative))
    expected = dict(id=variant["variant_id"], parent_episode_id=record["id"],
                    image_sha256=record["image_sha256"], prompt_family=family,
                    query=query, boxes=boxes, positive=positive, negative=negative,
                    source_id=variant["source_id"], prompt=prompt,
                    prompt_sha256=_sha(prompt.encode("utf-8")))
    if not approval or approval.get("decision") != APPROVED or approval.get("reviewer") != "Codex":
        raise ValueError("Exact runtime prompt has not been independently approved")
    for key, value in expected.items():
        if approval.get(key) != value:
            raise ValueError(f"Runtime review mismatch for {variant['variant_id']}: {key}")
    return dict(
        id=variant["variant_id"], parent_episode_id=record["id"], image=record["image"],
        image_sha256=record["image_sha256"], split="train", kind=kind, query=query,
        boxes=copy.deepcopy(boxes), positive=positive, negative=negative,
        prompt_family=family, prompt=prompt, prompt_sha256=expected["prompt_sha256"],
        label_basis=LABEL_BASIS, annotation_origin=variant["annotation_origin"],
        benchmark_dedup_status=episode["benchmark_dedup_status"], engineering_only=True,
        source_variant=variant["variant_id"], why=variant["why"],
        source_provenance=dict(candidate_file="correction_candidates/comparisons_v0.json",
                               source_id=variant["source_id"],
                               positive_step_id=variant["positive_step_id"],
                               positive_atom_index=variant["positive_atom_index"],
                               positive_episodes_file="provenance/source_positive_episodes.jsonl"),
        crop_files=copy.deepcopy(record["sources"][variant["source_id"]]["crop_files"]),
        runtime_prompt_review="correction_candidates/runtime_prompt_review_v1.json",
        target_scope="category_only" if kind == "binding" else "attribute_value",
        coordinate_exposure="exact_listed_boxes_in_runtime_text_and_resampler_only",
        runtime_audit=None, actual_student_error=False, actual_student_repair=False,
    )


def _asset_refs(value):
    if isinstance(value, str) and value.startswith(("images/", "inspection/")):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _asset_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _asset_refs(child)


def package(root, positive_root, output, runtime_review=None):
    root, positive_root, output = [Path(p).resolve() for p in (root, positive_root, output)]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen data version: {output}")
    if any(source == output or source.is_relative_to(output) for source in (root, positive_root)):
        raise ValueError("Output cannot contain a source directory")
    candidate_path = _inside(root, "correction_candidates/comparisons_v0.json")
    review_path = Path(runtime_review).resolve() if runtime_review else _inside(
        root, "correction_candidates/runtime_prompt_review_v1.json")
    candidate_bytes, review_bytes = candidate_path.read_bytes(), review_path.read_bytes()
    candidate, runtime = json.loads(candidate_bytes), json.loads(review_bytes)
    if candidate.get("label_basis") != LABEL_BASIS or runtime.get("label_basis") != LABEL_BASIS:
        raise ValueError("Unrecognized correction supervision provenance")
    if runtime.get("source_candidate_sha256") != _sha(candidate_bytes):
        raise ValueError("Runtime review was approved for a different candidate artifact")
    approvals = {row["id"]: row for row in runtime["approvals"]}
    if len(approvals) != len(runtime["approvals"]):
        raise ValueError("Duplicate runtime prompt approval")
    episodes_path = positive_root / "reviewed_episodes.jsonl"
    episodes = _lines(episodes_path)
    raw = _lines(root / "raw_tasks.jsonl")
    source_review_path = _inside(root, candidate["source_review_file"])
    original_reviews = {row["id"]: row for row in json.loads(source_review_path.read_bytes())}
    artifacts = {
        "correction_candidates/comparisons_v0.json": candidate_bytes,
        "correction_candidates/runtime_prompt_review_v1.json": review_bytes,
        "provenance/original_comparison_reviews.json": source_review_path.read_bytes(),
        "provenance/positive_version_manifest.json": (positive_root / "manifest.json").read_bytes(),
        "provenance/candidate_version_manifest.json": (root / "manifest.json").read_bytes(),
        "provenance/runtime_format.py": (REPO / "efa/format.py").read_bytes(),
        "provenance/package_codex_corrections.py": Path(__file__).read_bytes(),
    }
    pairs, withheld, selected_episodes, selected_raw, selected_reviews = [], [], [], [], []
    image_hashes, record_ids, variant_ids, asset_paths = set(), set(), set(), set()
    source_lines, pixel_checks = [], 0
    for record in candidate["records"]:
        rid = record["id"]
        if rid in record_ids or record["image_sha256"] in image_hashes:
            raise ValueError("Duplicate parent episode or image group")
        record_ids.add(rid)
        image_hashes.add(record["image_sha256"])
        episode, episode_line, line_number = episodes[rid]
        raw_row, raw_line, _ = raw[rid]
        original_review = original_reviews[rid]
        if episode.get("annotation_status") != "reviewed_episode":
            raise ValueError("Source positive episode was not explicitly reviewed")
        for row in (record, episode, raw_row, original_review):
            if row.get("split") != "train" or row["image_sha256"] != record["image_sha256"]:
                raise ValueError("Source image group or split does not match reviewed train positive")
        if partition(record["image_sha256"]) != "train":
            raise ValueError("Image-hash partition is not train")
        image_path = _inside(root, record["image"])
        image_bytes = image_path.read_bytes()
        if image_bytes != _inside(positive_root, episode["image"]).read_bytes():
            raise ValueError("Original image bytes differ from the frozen positive version")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        if (image.size != (episode["width"], episode["height"])
                or _sha(image.tobytes() + str(image.size).encode()) != record["image_sha256"]
                or _sha(image_bytes) != episode["image_file_sha256"]):
            raise ValueError("Original image dimensions, pixel hash or file hash mismatch")
        for atom in record["positive_atoms"]:
            expected = _atom(episode, atom["step_id"], atom["atom_index"])
            review_atom = _atom(original_review, atom["step_id"], atom["atom_index"])
            for key in ("subject", "predicate", "value"):
                if atom[key] != expected[key] or atom[key] != review_atom[key]:
                    raise ValueError("Candidate positive differs from frozen/reviewed positive atom")
            if record["sources"][atom["source_id"]]["boxes"] != expected["region"]["boxes"]:
                raise ValueError("Candidate positive boxes differ from frozen positive atom")
        for source in record["sources"].values():
            _boxes(source["boxes"], *image.size)
            if len(source["boxes"]) != len(source["crop_files"]):
                raise ValueError("Source boxes and crop files have different lengths")
            for box, crop_path in zip(source["boxes"], source["crop_files"]):
                with Image.open(_inside(root, crop_path)) as opened:
                    actual = opened.convert("RGB")
                expected = image.crop(box)
                if actual.size != expected.size or actual.tobytes() != expected.tobytes():
                    raise ValueError(f"Reviewed crop pixels differ from original crop: {crop_path}")
                pixel_checks += 1
        for variant in record["variants"]:
            vid = variant["variant_id"]
            if vid in variant_ids or variant.get("runtime_audit") is not None:
                raise ValueError("Duplicate variant or fabricated runtime audit")
            variant_ids.add(vid)
            pair = build_pair(record, variant, episode, approvals.get(vid))
            if pair is None:
                if vid in approvals or variant["labels"].get("binding") != "insufficient":
                    raise ValueError("Insufficient candidate has a runtime approval or invalid label")
                withheld.append(dict(
                    id=vid, parent_episode_id=rid, image=record["image"],
                    image_sha256=record["image_sha256"], split="train", training_eligible=False,
                    prompt_family=None, runtime_prompt=None, runtime_coordinate_approval=False,
                    source_variant=copy.deepcopy(variant),
                    source=copy.deepcopy(record["sources"][variant["source_id"]]),
                    original_exposure_policy=copy.deepcopy(candidate["exposure_policy"]),
                    label_basis=LABEL_BASIS, benchmark_dedup_status=episode["benchmark_dedup_status"],
                    runtime_audit=None,
                ))
            else:
                pair["image_file_sha256"] = _sha(image_bytes)
                pairs.append(pair)
        selected_episodes.append(episode_line)
        selected_raw.append(raw_line)
        selected_reviews.append(original_review)
        source_lines.append(dict(parent_episode_id=rid, original_line_number=line_number,
                                 original_line_sha256=_sha(episode_line)))
        asset_paths.update(_asset_refs(record))
        asset_paths.update(_asset_refs(original_review))
    if (len(pairs) != 16 or Counter(row["kind"] for row in pairs) != {"attribute": 12, "binding": 4}
            or len(withheld) != 2 or len(record_ids) != 6):
        raise ValueError("Version v1 requires exactly 16 reviewed pairs and 2 withheld candidates across 6 images")
    if set(approvals) != {row["id"] for row in pairs}:
        raise ValueError("Runtime approvals do not exactly match training pair IDs")
    if set(runtime["excluded_variant_ids"]) != {row["id"] for row in withheld}:
        raise ValueError("Insufficient exclusions do not match the runtime review")
    for relative in sorted(asset_paths):
        artifacts[relative] = _inside(root, relative).read_bytes()
    for name in ("render_new_views.py", "check_annotations.py"):
        relative = "inspection/comparison_errors/" + name
        artifacts[relative] = _inside(root, relative).read_bytes()
    artifacts.update({
        "correction_pairs.jsonl": _jsonl_bytes(pairs),
        "insufficient_candidates.jsonl": _jsonl_bytes(withheld),
        "provenance/source_positive_episodes.jsonl": b"".join(selected_episodes),
        "provenance/source_raw_tasks.jsonl": b"".join(selected_raw),
        "provenance/source_positive_line_provenance.json": _json_bytes(source_lines),
        "codex_reviews/comparisons.json": _json_bytes(selected_reviews),
        "README.md": ("# Codex correction pairs v1\n\n"
            "16 constructed, visually reviewed contrasts: 12 attribute pairs and 4 object-category binding pairs. "
            "All 6 source images are train groups. This is engineering-only data with benchmark deduplication pending.\n\n"
            "Use correction_pairs.jsonl only. Its prompt matches efa.format.read_prompt or binding_prompt exactly. "
            "For attribute reads, query is the original positive subject/predicate; for binding, query is null and "
            "only actual/impostor object categories are suffix targets. Binding pairs must never apply color UL. "
            "Only suffix sequences are CE/UL targets; queries, metadata and corrupted drafts are not positive labels.\n\n"
            "correction_candidates/runtime_prompt_review_v1.json separately approves coordinate text and "
            "same-box resampler coordinates for exactly these 16 contrasts. Binding supervision is category-only, "
            "not full spatial/ownership verification. No additional full-frame/context pixels or reasoning cache are exposed.\n\n"
            "The 2 insufficient_candidates.jsonl entries are NOT training pairs and retain their earlier "
            "pixels-only/no-coordinate constraint. Do not reinterpret their original facts or colors as false.\n\n"
            "Original candidate JSON, exact source positive lines, original image bytes, reviewed native crops "
            "and inspection panels are retained. provenance/original_comparison_reviews.json is the full original "
            "review file for provenance only; codex_reviews/comparisons.json selects the six relevant records. "
            "Inspection scripts are preserved as source-workspace provenance, not standalone entry points.\n\n"
            "No actual student errors, repairs, runtime gate scores or human-gold labels are claimed. "
            "The data has not been uploaded or added to an active training run by this packager.\n").encode("utf-8"),
    })
    manifest = dict(
        schema="efa.codex_correction_pairs.v1", kind="reviewed_constructed_semantic_contrasts",
        count=len(pairs), training_pairs=len(pairs),
        pair_kind_counts=dict(Counter(row["kind"] for row in pairs)),
        insufficient_candidates=len(withheld), source_positive_episodes=len(record_ids),
        images=len(image_hashes), train_image_groups=len(image_hashes), split_counts={"train": len(pairs)},
        exact_crop_pixel_checks=pixel_checks, native_object_ids_generated=False,
        original_images_edited=False, original_image_bytes_preserved=True,
        label_basis=LABEL_BASIS, engineering_only=True, benchmark_dedup_status="pending",
        runtime_audits=0, actual_student_errors=0, actual_student_repairs=0, human_gold_records=0,
        runtime_coordinate_approved_pairs=16, insufficient_runtime_coordinate_approvals=0,
        runtime_prompt_review="correction_candidates/runtime_prompt_review_v1.json",
        provenance_policy="Exact positive source rows and original artifacts retained; recorded visual review is not measured runtime verification.",
        source_candidate_sha256=_sha(candidate_bytes), source_runtime_review_sha256=_sha(review_bytes),
        source_positive_episodes_file_sha256=_sha(episodes_path.read_bytes()),
        files_sha256={name: _sha(content) for name, content in sorted(artifacts.items())},
        checksum_scope="files_sha256 covers all payloads. SHA256SUMS also covers manifest.json and excludes only itself.",
        construction_complete=True,
    )
    artifacts["manifest.json"] = _json_bytes(manifest)
    artifacts["SHA256SUMS"] = "".join(
        f"{_sha(content)}  {name}\n" for name, content in sorted(artifacts.items())
    ).encode("utf-8")
    output.mkdir(parents=True, exist_ok=False)
    for name, content in artifacts.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        if _sha(destination.read_bytes()) != _sha(content):
            raise RuntimeError(f"Written artifact hash mismatch: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/deepeyesv2_curated48_candidates_v1")
    parser.add_argument("--positive-root", default="data/codex_curated45_adjudicated_v1")
    parser.add_argument("--runtime-review")
    parser.add_argument("--output", default="data/codex_correction_pairs_v1")
    args = parser.parse_args()
    result = package(args.root, args.positive_root, args.output, args.runtime_review)
    print(json.dumps({key: value for key, value in result.items() if key != "files_sha256"}, indent=2))


if __name__ == "__main__":
    main()
