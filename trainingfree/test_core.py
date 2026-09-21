import unittest
from PIL import Image
from .core import Audit, Step, parse_json, solve
from .online import Online, control_boxes, mask, overlap
from .run import extract_answer


class Scripted:
    def __init__(self, steps, audits):
        self.steps, self.audits = iter(steps), iter(audits)
        self.contexts = []

    def propose(self, image, question, ledger, feedback):
        self.contexts.append(([s.claim for s in ledger], feedback))
        step = next(self.steps)
        if isinstance(step, Exception):
            raise step
        return step

    def audit(self, image, question, step, ledger):
        return Audit(next(self.audits))


class Tests(unittest.TestCase):
    def test_repair_does_not_pollute_next_context(self):
        b = Scripted([Step('wrong'), Step('right'), Step('therefore', 'deduction', final_answer='A')],
                     ['contradicted', 'supported', 'supported'])
        result = solve(b, None, 'question')
        self.assertEqual(result['answer'], 'A')
        self.assertEqual(b.contexts[2], (['right'], ''))
        self.assertEqual(result['trace'][2]['parents'], [0])

    def test_no_silent_pass_on_exhaustion(self):
        b = Scripted([Step('ambiguous')]*3, ['inconclusive']*3)
        result = solve(b, None, 'q')
        self.assertIsNone(result['answer'])
        self.assertEqual(result['ledger'], [])
        self.assertEqual(result['status'], 'verification_exhausted')

    def test_first_step_deduction_cannot_bypass_image(self):
        b = Scripted([Step('answer', 'deduction', final_answer='A')], [])
        self.assertIsNone(solve(b, None, 'q', repairs=0)['answer'])

    def test_parse_error_is_bounded(self):
        b = Scripted([ValueError('bad')]*2, [])
        self.assertEqual(solve(b, None, 'q', repairs=1)['status'], 'verification_exhausted')

    def test_step_limit(self):
        b = Scripted([Step('fact')], ['supported'])
        self.assertEqual(solve(b, None, 'q', max_steps=1)['status'], 'step_budget_exhausted')

    def test_json_boxes(self):
        self.assertEqual(parse_json('```json\n{"a":1}\n```'), {'a': 1})
        for box in ([0,0,0,1], [0,0,2,1], [float('nan'),0,1,1], [0,0,1]):
            with self.assertRaises(ValueError):
                Step.from_dict({'claim':'x', 'box': box})

    def test_controls_area_and_overlap(self):
        box = (.2,.2,.4,.4)
        controls = control_boxes(box)
        self.assertEqual(len(controls), 2)
        for c in controls:
            self.assertAlmostEqual((c[2]-c[0])*(c[3]-c[1]), .04)
            self.assertLessEqual(overlap(box, c), .1)

    def test_mask_does_not_mutate_image(self):
        image = Image.new('RGB', (20,20), (255,0,0))
        image.putpixel((4,4), (0,0,255))
        changed = mask(image, (.1,.1,.4,.4))
        self.assertEqual(image.getpixel((4,4)), (0,0,255))
        self.assertNotEqual(changed.getpixel((4,4)), image.getpixel((4,4)))

    def test_visual_final_requires_deduction(self):
        audit = Online(None).audit(None, 'q', Step('fact', box=(0,0,.2,.2), final_answer='A'), [])
        self.assertNotEqual(audit.status, 'supported')

    def test_answer_extraction(self):
        self.assertEqual(extract_answer('reason\nFINAL ANSWER: B.'), 'B')
        self.assertEqual(extract_answer('Thus, Yes.'), 'Yes')
        self.assertEqual(extract_answer('Thus, the answer is **A**.'), 'A')
        self.assertEqual(extract_answer(None), '')
        self.assertEqual(extract_answer('B. A blue car.', 'A. red B. blue'), 'B')


if __name__ == '__main__':
    unittest.main()
