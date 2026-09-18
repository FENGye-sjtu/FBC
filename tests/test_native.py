import unittest
from efa.native import property_atom
try:
    from efa.zwz import canonical_qa
except ImportError:
    canonical_qa=None


class NativeTests(unittest.TestCase):
    @unittest.skipIf(canonical_qa is None,'Optional data extra is unavailable')
    def test_native_relation_answers_are_not_discarded(self):
        q,v=canonical_qa(dict(answer='in front of',extra_info=dict(question='Where is the pool relative to the building?\nPlease identify their relative position: left of / right of.')))
        self.assertEqual(v,'in front of')
        self.assertEqual(q,'Where is the pool relative to the building?')

    def test_neutral_property_preserves_referent(self):
        a=property_atom('What color is the lampshade?','beige')
        self.assertEqual(a['subject'],'the lampshade')
        self.assertEqual(a['predicate'],'color')
        self.assertEqual(a['value'],'beige')

    def test_unsupported_or_contaminated_queries_are_not_invented(self):
        self.assertIsNone(property_atom('What color is the red bag?','red'))
        self.assertIsNone(property_atom('Why did the person leave?','unknown'))


if __name__=='__main__':
    unittest.main()
