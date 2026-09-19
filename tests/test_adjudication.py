"""Packager acceptance boundaries, provenance and actual exporter integration."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.package_adjudication import LABEL_BASIS, package
from efa.export import export


class AdjudicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "source"
        (self.root / "images").mkdir(parents=True)
        (self.root / "codex_reviews").mkdir()
        (self.root / "teacher_cache").mkdir()
        image = Image.new("RGB", (16, 12), (10, 30, 50))
        image.save(self.root / "images/source.png")
        sha = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
        self.raw = dict(id="fixture", image="images/source.png", image_sha256=sha,
            width=16, height=12, question="How many objects?", answer="1", split="train",
            source_repo="synthetic/unit-test", source_revision="fixture", benchmark_dedup_status="pending",
            annotation_status="source_qa_requires_episode_conversion", label_basis="synthetic_test_fixture")
        self.review = dict(id="fixture", reviewer="codex_visual_annotator", image_sha256=sha,
            decision="revise", answer="2", visual_findings=["Synthetic fixture, not a real review."],
            step_count_reason="Direct count fixture.", views_inspected=["images/source.png"],
            steps=[dict(id=1, text="Step 1: There are 2 objects.", parents=[], atoms=[dict(
                subject="all visible objects", predicate="count", value="2", family="count", kind="visual",
                region=dict(boxes=[[0,0,16,12]], context_boxes=[], insufficient=False,
                    coordinate_system="original_pixels"))])])
        self.write_inputs()

    def write_inputs(self, reviews=None):
        (self.root / "raw_tasks.jsonl").write_text(json.dumps(self.raw) + "\n", encoding="utf-8")
        (self.root / "codex_reviews/review.json").write_text(json.dumps(reviews or [self.review]), encoding="utf-8")
        (self.root / "teacher_draft_results.jsonl").write_text(
            json.dumps(dict(id="fixture", error="fixture teacher failed")) + "\n", encoding="utf-8")
        (self.root / "teacher_cache/request.json").write_text('{"output": {"fixture": true}}', encoding="utf-8")

    def test_explicit_correction_is_preserved_and_exports_only_with_override(self):
        output = self.base / "v1"
        manifest = package(self.root, output)
        episode = json.loads((output / "reviewed_episodes.jsonl").read_text())
        self.assertEqual((episode["source_answer"], episode["answer"]), ("1", "2"))
        self.assertEqual(episode["label_basis"], LABEL_BASIS)
        self.assertEqual(episode["codex_review"]["decision"], "revise")
        self.assertIsNone(episode["runtime_audit"])
        self.assertEqual(episode['steps'][0]['atoms'][0]['region']['context_boxes'], [[0,0,16,12]])
        self.assertEqual(manifest["source_answer_string_changes"], 1)
        self.assertTrue(manifest["engineering_only"])
        for name in ("raw_tasks.jsonl", "teacher_draft_results.jsonl", "codex_reviews/review.json", "teacher_cache/request.json", "images/source.png"):
            self.assertEqual((output / name).read_bytes(), (self.root / name).read_bytes())
        export(output, self.base / "formal")
        self.assertEqual(json.loads((self.base / "formal/efa_steps.json").read_text()), [])
        export(output, self.base / "engineering", allow_unchecked=True)
        self.assertEqual(len(json.loads((self.base / "engineering/efa_steps.json").read_text())), 1)
        route = json.loads(json.loads((self.base / "engineering/efa_regions.json").read_text())[0]['messages'][1]['content'])
        self.assertEqual(route['context_boxes'], route['boxes'])
        with self.assertRaises(FileExistsError):
            package(self.root, output)

    def test_needs_review_is_retained_but_never_exported_even_when_answer_matches(self):
        self.review.update(decision="needs_review", answer=self.raw["answer"], steps=[])
        self.write_inputs()
        output = self.base / "withheld"
        manifest = package(self.root, output)
        self.assertEqual(manifest["complete_efa_episodes"], 0)
        self.assertEqual(manifest["decisions"], {"needs_review": 1})
        self.assertEqual((output / "reviewed_episodes.jsonl").read_text(), "")
        self.assertEqual(json.loads((output / "adjudication_records.jsonl").read_text())["decision"], "needs_review")

    def test_invalid_regions_hash_answer_or_evidence_do_not_create_version(self):
        baseline = copy.deepcopy(self.review)
        mutations = [
            lambda row: row["steps"][0]["atoms"][0]["region"].update(boxes=[[0,0,17,12]]),
            lambda row: row.update(image_sha256="0" * 64),
            lambda row: row.update(answer="3"),
            lambda row: row.update(views_inspected=[]),
            lambda row: row.update(split="dev"),
            lambda row: row["steps"][0]["atoms"][0]["region"].update(insufficient=True),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                self.review = copy.deepcopy(baseline)
                mutate(self.review)
                self.write_inputs()
                output = self.base / f"invalid{index}"
                with self.assertRaises(ValueError):
                    package(self.root, output)
                self.assertFalse(output.exists())
        self.review = baseline
        self.write_inputs()
        Image.new("RGB", (16,12), "red").save(self.root / "images/source.png")
        with self.assertRaisesRegex(ValueError, "pixel hash"):
            package(self.root, self.base / "changed_pixels")

    def test_rewrite_requires_explicit_provenance_and_legacy_view_evidence_works(self):
        self.review["question"] = "How many visible objects are in the entire image?"
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "explicitly marked"):
            package(self.root, self.base / "unmarked")
        self.review.update(question_revised=True, question_revision_reason="Make the exhaustive extent explicit.")
        self.review.pop("views_inspected")
        self.review["source_checks"] = dict(visual_review_evidence=["original image"])
        self.write_inputs()
        output = self.base / "rewritten"
        manifest = package(self.root, output)
        episode = json.loads((output / "reviewed_episodes.jsonl").read_text())
        self.assertEqual(episode["source_question"], self.raw["question"])
        self.assertEqual(episode["question_revision"]["revised"], self.review["question"])
        self.assertEqual(manifest["question_rewrites"], 1)


if __name__ == "__main__":
    unittest.main()
