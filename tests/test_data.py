import unittest
from efa.regions import normalized_boxes
from efa.annotate import validate_steps
from efa.review import usable_sources


class AnnotationTests(unittest.TestCase):
    def test_insufficient_region_cannot_enter_training(self):
        atom={'kind':'visual','region':{'insufficient':True}}
        row={'steps':[{'atoms':[atom]}]}
        self.assertFalse(usable_sources(row))
        atom['region']['insufficient']=False
        self.assertTrue(usable_sources(row))

    def test_teacher_boxes_convert_into_original_pixel_frame(self):
        self.assertEqual(normalized_boxes([250,100,750,300],2000,1000),[[500,100,1500,300]])
        with self.assertRaises(ValueError):
            normalized_boxes([[0,0,1200,300]],2000,1000)

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
