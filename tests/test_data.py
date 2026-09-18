import unittest
from efa.annotate import validate_steps


class AnnotationTests(unittest.TestCase):
    def test_dependency_must_exist(self):
        with self.assertRaises(ValueError):
            validate_steps({"steps":[{"id":1,"text":"Step 1: blue","parents":[2],
                "atoms":[{"subject":"bag","predicate":"color","value":"blue","kind":"visual"}]}]})

    def test_empty_atom_cannot_pass(self):
        with self.assertRaises(ValueError):
            validate_steps({"steps":[{"id":1,"text":"Step 1: blue","parents":[],"atoms":[]}]})

    def test_premise_requires_parents(self):
        with self.assertRaises(ValueError):
            validate_steps({"steps":[{"id":1,"text":"Step 1: blue","parents":[],
                "atoms":[{"subject":"bag","predicate":"color","value":"blue","kind":"premise"}]}]})
