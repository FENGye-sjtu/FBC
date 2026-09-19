import json
import unittest
try:
    import lmformatenforcer
except ImportError:
    lmformatenforcer=None
from efa.grammar import step_parser,trajectory_step_parser
from efa.format import STEP_END


@unittest.skipIf(lmformatenforcer is None,'Optional grammar extra unavailable')
class GrammarTests(unittest.TestCase):
    def test_empty_marker_cannot_replace_an_atom(self):
        parser=step_parser()
        self.assertNotIn('<',parser.get_allowed_characters())
        content=json.dumps({'atoms':[dict(subject='bag',predicate='color',value='blue',family='attribute',kind='visual')],'final':True})
        for char in content:
            self.assertIn(char,parser.get_allowed_characters())
            parser=parser.add_character(char)
        self.assertFalse(parser.can_end())
        for char in STEP_END:
            self.assertIn(char,parser.get_allowed_characters())
            parser=parser.add_character(char)
        self.assertTrue(parser.can_end())

    def test_complete_trajectory_requires_numbered_visible_text(self):
        parser=trajectory_step_parser()
        content=json.dumps({'step':2,'text':'Compare the two verified counts.',
            'atoms':[dict(subject='counts',predicate='comparison',value='yes',family='deduction',kind='premise')],
            'final':True})+STEP_END
        for char in content:
            self.assertIn(char,parser.get_allowed_characters())
            parser=parser.add_character(char)
        self.assertTrue(parser.can_end())


if __name__=='__main__':
    unittest.main()
