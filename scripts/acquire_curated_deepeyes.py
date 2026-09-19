"""Read selected original DeepEyesV2 images from an existing local archive.

No downloads, source trace supervision, automatic approval or archive-wide
extraction. Existing image hashes are excluded before assigning project splits.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from efa.data import partition
from efa.deepeyes import REVISION


def _jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def _source_qa(row):
    users = [m["content"] for m in row["messages"] if m["role"] == "user"]
    assistants = [m["content"] for m in row["messages"] if m["role"] == "assistant"]
    if not users or not assistants or not isinstance(users[0], str) or users[0].count("<image>") != 1:
        raise ValueError("Expected original single-image user question")
    question = users[0].split("You must put your answer inside")[0].replace("<image>", "").strip()
    answers = re.findall(r"<answer>(.*?)</answer>", assistants[-1], re.S)
    if not question or not answers:
        raise ValueError("Source question or unverified final answer missing")
    has_external = any(
        m["role"] == "assistant" and ("<tool_call>" in m.get("content", "") or m.get("tool_calls"))
        for m in row["messages"]
    )
    return question, answers[-1].strip(), bool(has_external)


def _registry(root, candidate_hashes):
    """Record only matching hashes; do not retain unrelated project data in RAM."""
    matches, files = {}, []
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Registry directory does not exist: {root}")
    paths = sorted(set(root.glob("*/raw_tasks.jsonl")) | set(root.glob("*/native_episodes.jsonl")))
    total_records = 0
    for path in paths:
        digest = hashlib.sha256()
        records = 0
        with path.open("rb") as stream:
            for number, line in enumerate(stream, 1):
                digest.update(line)
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError) as error:
                    raise ValueError(f"Invalid registry record {path}:{number}") from error
                records += 1
                image_sha = record.get("image_sha256")
                if image_sha in candidate_hashes:
                    matches.setdefault(image_sha, []).append(dict(
                        registry_file=path.relative_to(root).as_posix(), id=record.get("id"),
                        split=record.get("split"), source_repo=record.get("source_repo")))
        files.append(dict(path=path.relative_to(root).as_posix(), sha256=digest.hexdigest(), records=records))
        total_records += records
    return matches, dict(root=str(root), files=files, records_scanned=total_records,
        policy="exclude_matching_decoded_pixel_hashes; never reassign_existing_image_groups",
        missing_hash_policy="records_without_image_sha256_cannot_establish_pixel_match",
        near_duplicate_status="pending")


def acquire(selection_path, source, output, registry_root=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    selection_path = Path(selection_path).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output version: {output}")
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    if selection.get("source_revision") != REVISION:
        raise ValueError("Selection revision differs from pinned DeepEyesV2 source")
    candidates = selection["candidates"]
    if selection.get("candidate_count") != len(candidates):
        raise ValueError("Selection candidate count mismatch")
    metadata_path = source / "json/sft_part_0.json"
    metadata_bytes = metadata_path.read_bytes()
    metadata_sha = hashlib.sha256(metadata_bytes).hexdigest()
    if metadata_sha != selection.get("source_sha256"):
        raise ValueError("Pinned source metadata SHA256 mismatch")
    rows = json.loads(metadata_bytes)
    archive_path = source / "images.zip"
    archive_stat = archive_path.stat()
    prepared, blobs, ids = [], {}, set()
    with zipfile.ZipFile(archive_path) as archive:
        names = Counter(info.filename for info in archive.infolist())
        for candidate in candidates:
            index = candidate["source_row"]
            if type(index) is not int or not 0 <= index < len(rows):
                raise ValueError("Invalid selected source row")
            rid = f"deepeyesv2-p0-{index}"
            if candidate["id"] != rid or rid in ids:
                raise ValueError("Selection ID/row mismatch or duplicate row")
            ids.add(rid)
            if candidate.get("source_repo") != "honglyhly/DeepEyesV2_SFT" or candidate.get("source_revision") != REVISION:
                raise ValueError(f"{rid}: source repository/revision mismatch")
            if candidate.get("source_metadata_file") != "json/sft_part_0.json":
                raise ValueError(f"{rid}: unexpected metadata file")
            row = rows[index]
            original_member = row["images"][0]
            if original_member != candidate["source_image"]:
                raise ValueError(f"{rid}: original image differs from selected source member")
            question, answer, external = _source_qa(row)
            if question != candidate["question"] or answer != str(candidate["upstream_answer_unverified"]):
                raise ValueError(f"{rid}: original QA differs from selected metadata")
            exception = index == 422 and candidate.get("existing_builder_compatible") is False
            if external and not exception:
                raise ValueError(f"{rid}: external-tool source requires an explicit extraction-only exception")
            if exception and candidate.get("candidate_type") != "chart_candidate_requires_independent_reannotation":
                raise ValueError("Row 422 exception must be marked for independent chart reannotation")
            possible = list(dict.fromkeys([original_member, original_member.removeprefix("images/")]))
            if candidate.get("zip_member_candidates") != possible:
                raise ValueError(f"{rid}: archive member candidate list mismatch")
            found = [name for name in possible if name in names]
            if len(found) != 1 or names[found[0]] != 1:
                raise ValueError(f"{rid}: missing or ambiguous original archive member")
            info = archive.getinfo(found[0])
            if info.is_dir():
                raise ValueError("Selected archive member is a directory")
            content = archive.read(info)  # Reads this member only and verifies its ZIP CRC.
            with Image.open(io.BytesIO(content)) as decoded:
                image_format = decoded.format
                image = decoded.convert("RGB")
            extensions = {"JPEG":"jpg", "PNG":"png", "WEBP":"webp", "GIF":"gif", "BMP":"bmp", "TIFF":"tiff"}
            if image_format not in extensions:
                raise ValueError(f"Unsupported original image format: {image_format}")
            sha = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
            relative = f"images/{sha}.{extensions[image_format]}"
            # No save/re-encode: original compressed image payload is retained byte-for-byte.
            blobs.setdefault(relative, content)
            prepared.append(dict(
                id=rid, source_repo="honglyhly/DeepEyesV2_SFT", source_revision=REVISION,
                source_metadata_file="json/sft_part_0.json", source_metadata_sha256=metadata_sha,
                source_row=index, source_image=original_member,
                source_split=row.get("split", "not_provided_in_source_metadata"),
                source_collection="upstream_sft_part_0; original benchmark provenance remains unverified",
                image=relative, image_sha256=sha, image_file_sha256=hashlib.sha256(content).hexdigest(),
                width=image.width, height=image.height, question=question, answer=answer,
                source_question=question, upstream_answer_unverified=answer,
                source_archive=dict(path="images.zip", archive_bytes=archive_stat.st_size,
                    member=info.filename, crc32=f"{info.CRC:08x}", file_size=info.file_size,
                    compress_size=info.compress_size, compress_type=info.compress_type, header_offset=info.header_offset),
                source_trace=[], source_trace_reused=False,
                source_trace_policy="No original assistant/tool trace is used as supervision; fresh image-grounded annotation required.",
                source_contains_external_tool_call=external,
                extraction_only_reannotation_exception=exception,
                extraction_exception_note=("Row 422: original image only; external tool trace discarded. Chart sufficiency and answer require independent visual adjudication." if exception else None),
                selected_original_image_index=0, other_source_images_not_extracted=len(row["images"])-1,
                candidate_type=candidate["candidate_type"], selection_record=candidate,
                split=partition(sha), split_basis="efa.data.partition(decoded_RGB_bytes_plus_size_sha256)",
                annotation_status="source_qa_requires_episode_conversion",
                benchmark_dedup_status="pending", image_quality_status="unseen",
                label_basis="upstream_teacher_answer_not_ground_truth", steps=None, runtime_audit=None,
            ))
    registry_root = Path(registry_root).resolve() if registry_root else output.parent
    existing, registry = _registry(registry_root, {row["image_sha256"] for row in prepared})
    accepted, exclusions, selected_hashes = [], [], {}
    for row in prepared:
        sha = row["image_sha256"]
        if sha in existing:
            exclusions.append(dict(id=row["id"], reason="existing_image_hash", image_sha256=sha,
                registry_matches=existing[sha], selection_record=row["selection_record"],
                source_archive=row["source_archive"]))
        elif sha in selected_hashes:
            exclusions.append(dict(id=row["id"], reason="candidate_duplicate_image_hash", image_sha256=sha,
                kept_candidate_id=selected_hashes[sha], selection_record=row["selection_record"],
                source_archive=row["source_archive"]))
        else:
            selected_hashes[sha] = row["id"]
            accepted.append(row)
    after = archive_path.stat()
    if (archive_stat.st_size, archive_stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Source archive changed during selected extraction")
    manifest = dict(kind="curated_original_image_candidates_not_approved", count=len(accepted),
        selected_metadata_candidates=len(candidates), excluded=len(exclusions),
        exclusion_reasons=dict(Counter(row["reason"] for row in exclusions)),
        categories=dict(Counter(row["candidate_type"] for row in accepted)),
        split_counts=dict(Counter(row["split"] for row in accepted)),
        source_repo="honglyhly/DeepEyesV2_SFT", source_revision=REVISION,
        source_metadata_sha256=metadata_sha, selection_sha256=hashlib.sha256(selection_bytes).hexdigest(),
        source_archive="images.zip", source_archive_bytes=archive_stat.st_size,
        selected_archive_members_read=len(prepared), archive_wide_extraction=False,
        original_images_copied=len(accepted), image_edits=False, original_image_bytes_preserved=True,
        source_trace_supervision=False, external_trace_extraction_exceptions=sum(row["extraction_only_reannotation_exception"] for row in accepted),
        source_split_note="Project image-hash split is separate from unknown original benchmark split.",
        complete_efa_episodes=0, reviewed=0, visual_reviews=0, runtime_audited_episodes=0,
        annotation_status="pending_individual_visual_adjudication", benchmark_dedup_status="pending",
        engineering_only=True, label_basis="upstream_teacher_answer_not_ground_truth",
        registry_files_scanned=len(registry["files"]), registry_records_scanned=registry["records_scanned"],
        construction_complete=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    for row in accepted:
        (output / row["image"]).write_bytes(blobs[row["image"]])
    (output / "selection.json").write_bytes(selection_bytes)
    (output / "raw_tasks.jsonl").write_text(_jsonl(accepted), encoding="utf-8")
    (output / "exclusions.jsonl").write_text(_jsonl(exclusions), encoding="utf-8")
    (output / "hash_registry_scan.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--registry-root", help="Existing derived dataset directory; defaults to output parent")
    args = parser.parse_args()
    print(json.dumps(acquire(args.selection, args.source, args.output, args.registry_root), ensure_ascii=False))


if __name__ == "__main__":
    main()
