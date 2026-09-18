import unittest
from efa.regions import normalized_boxes
from efa.annotate import validate_steps
from efa.review import usable_sources


class AnnotationTests(unittest.TestCase):
    def test_bare_relation_is_not_an_independent_query(self):
        atom={'subject':'person','predicate':'relation','value':'left','family':'relation',
              'kind':'visual','region':{'insufficient':False}}
        row={'steps':[{'id':1,'text':'Step 1: Person is left of the sign.','parents':[],'atoms':[atom]}]}
        self.assertFalse(usable_sources(row))
        with self.assertRaises(ValueError):validate_steps(row)

    def test_teacher_serialization_normalization_preserves_original(self):
        original={'steps':[{'id':1,'text':'Count visible bags.','parents':[],
            'atoms':[{'subject':'bags','predicate':'count','value':2,'family':'count','kind':'visual'}]}]}
        normalized=validate_steps(original)
        self.assertEqual(normalized[0]['text'],'Step 1: Count visible bags.')
        self.assertEqual(normalized[0]['atoms'][0]['value'],'2')
        self.assertEqual(original['steps'][0]['atoms'][0]['value'],2)

    def test_deduction_alias_still_requires_explicit_parents(self):
        row={'steps':[{'id':1,'text':'Compute total.','parents':[],
            'atoms':[{'subject':'total','predicate':'sum','value':2,'family':'numeric','kind':'deduction'}]}]}
        with self.assertRaises(ValueError):validate_steps(row)

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
