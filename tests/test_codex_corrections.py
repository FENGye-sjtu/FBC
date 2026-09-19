"""Semantic boundaries for exporting reviewed correction supervision."""
import copy
import hashlib
import re
import unittest

from efa.format import binding_prompt, read_prompt
from scripts.package_codex_corrections import APPROVED, build_pair


class CodexCorrectionTests(unittest.TestCase):
    def fixture(self, kind):
        episode = dict(id="example", split="train", image_sha256="a" * 64,
                       benchmark_dedup_status="pending", steps=[
            dict(id=1, atoms=[dict(subject="the shirt", predicate="main color",
                                 value="yellow", region=dict(boxes=[[0, 0, 8, 8]]))]),
            dict(id=2, atoms=[dict(subject="the skirt", predicate="main color",
                                 value="yellow", region=dict(boxes=[[0, 8, 8, 16]]))]),
        ])
        record = dict(id=episode["id"], split="train", image="images/example.png",
                      image_sha256=episode["image_sha256"], sources={
            "s1": dict(boxes=[[0, 0, 8, 8]], crop_files=["inspection/shirt.png"]),
            "s2": dict(boxes=[[0, 8, 8, 16]], crop_files=["inspection/skirt.png"]),
        })
        variant = dict(variant_id="fixture", positive_step_id=1, positive_atom_index=0,
                       annotation_origin="synthetic_unit_test_fixture_not_real_annotation",
                       why="Unit-test geometry and labels only.", runtime_audit=None)
        if kind == "attribute":
            variant.update(type="correct_source_wrong_attribute", source_id="s1",
                           training_eligibility="attribute_ce_ul", incumbent_value="red",
                           corrected_value="yellow", labels=dict(attribute="contradicted", binding="not_supervised"))
            query = dict(subject="the shirt", predicate="main color")
            positive, negative, family = "yellow", "red", "runtime_read"
            prompt = read_prompt(query, record["sources"]["s1"]["boxes"])
        else:
            variant.update(type="wrong_object_same_attribute", source_id="s2",
                           source_from_positive_step_id=2, training_eligibility="binding_ce_ul_only",
                           incumbent_value="yellow", actual_object_identity="skirt",
                           impersonated_object_identity="shirt",
                           labels=dict(binding="contradicted", local_attribute="supported"))
            query, positive, negative, family = None, "skirt", "shirt", "runtime_binding"
            prompt = binding_prompt(record["sources"]["s2"]["boxes"])
        approval = dict(id=variant["variant_id"], parent_episode_id=record["id"],
                        image_sha256=record["image_sha256"], prompt_family=family,
                        query=query, boxes=record["sources"][variant["source_id"]]["boxes"],
                        positive=positive, negative=negative, source_id=variant["source_id"],
                        prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                        decision=APPROVED, reviewer="Codex")
        return record, variant, episode, approval

    def test_binding_targets_are_categories_and_query_is_null(self):
        record, variant, episode, approval = self.fixture("binding")
        pair = build_pair(record, variant, episode, approval)
        self.assertEqual((pair["positive"], pair["negative"]), ("skirt", "shirt"))
        self.assertIsNone(pair["query"])
        self.assertEqual(pair["prompt_family"], "runtime_binding")
        self.assertEqual(pair["prompt"], binding_prompt(pair["boxes"]))
        self.assertNotIn("yellow", pair["prompt"])
        for bad_color in ("yellow", "red"):
            bad_variant = dict(variant, impersonated_object_identity=bad_color)
            with self.assertRaisesRegex(ValueError, "never colors"):
                build_pair(record, bad_variant, episode, approval)

    def test_insufficient_is_excluded_and_cannot_be_promoted(self):
        record, variant, episode, approval = self.fixture("attribute")
        variant.update(type="missing_object_anchor", training_eligibility="insufficient_candidate_only")
        self.assertIsNone(build_pair(record, variant, episode, None))
        variant["training_eligibility"] = "attribute_ce_ul"
        with self.assertRaisesRegex(ValueError, "cannot be promoted"):
            build_pair(record, variant, episode, approval)

    def test_original_query_and_exact_coordinate_prompt_without_candidate_leakage(self):
        record, variant, episode, approval = self.fixture("attribute")
        pair = build_pair(record, variant, episode, approval)
        self.assertEqual(pair["query"], dict(subject="the shirt", predicate="main color"))
        self.assertEqual(pair["prompt"], read_prompt(pair["query"], pair["boxes"]))
        self.assertNotRegex(pair["prompt"], r"\b" + re.escape(pair["negative"]) + r"\b")
        for leaked_value in ("red", "yellow"):
            leaked = copy.deepcopy(episode)
            leaked["steps"][0]["atoms"][0]["subject"] = f"the {leaked_value} shirt"
            with self.assertRaisesRegex(ValueError, "leaked"):
                build_pair(record, variant, leaked, approval)
        unapproved = dict(approval, prompt="Same source, but coordinates were omitted")
        with self.assertRaisesRegex(ValueError, "Runtime review mismatch"):
            build_pair(record, variant, episode, unapproved)


if __name__ == "__main__":
    unittest.main()
