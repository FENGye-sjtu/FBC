import unittest

from trainingfree.workstreams.faithact75_20260923.isolated_bf16_queue import select_devices
from trainingfree.workstreams.faithact75_20260923.evidence_first_v22 import has_inference_language
from trainingfree.workstreams.faithact75_20260923.selective_faithact import choose


class SelectiveFaithActTest(unittest.TestCase):
    def test_accepts_only_unanimously_supported_valid_revision(self):
        anchor = {"answer": "A"}
        revision = {"answer": "B"}
        self.assertIs(choose(anchor, revision, [{"answer": "B"}, {"answer": "B"}])[0], revision)
        self.assertIs(choose(anchor, revision, [{"answer": "B"}, {"answer": "A"}])[0], anchor)
        self.assertIs(choose(anchor, {"answer": "unknown"}, [{"answer": "unknown"}])[0], anchor)

    def test_queue_requires_declared_capacity_after_reserve(self):
        self.assertEqual(select_devices([(0, 10000), (1, 9000)], 16000, 1500),
                         [(0, 10000), (1, 9000)])
        self.assertEqual(select_devices([(0, 10000)], 9000, 1500), [])

    def test_visual_inference_language_is_rejected(self):
        self.assertTrue(has_inference_language("The road is uphill because it rises."))
        self.assertTrue(has_inference_language("The left position indicates the same direction."))
        self.assertFalse(has_inference_language("The lane edges converge above the image center."))


if __name__ == "__main__":
    unittest.main()
