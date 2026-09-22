import unittest
from PIL import Image
from .core import Audit, Step, parse_json, solve
from .online import Online, control_boxes, mask, overlap
from .run import extract_answer
from .prompts import task_question, answer_format, cot_prompt
from .queue_bf16 import select_devices
from .placement import pack_modules
from .format_adapter import FinalFormatAdapter
from .qwen import align_rope_delta


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

    def finalize(self, image, question, ledger):
        return 'B'


class Tests(unittest.TestCase):
    def test_multigpu_rope_delta_follows_dispatched_cache(self):
        class Delta:
            def __init__(self):
                self.moved_to = None
            def to(self, device):
                self.moved_to = device
                return self
        model = type('Model', (), {'rope_deltas': Delta()})()
        cache = type('Cache', (), {'device': 'cuda:1'})()
        align_rope_delta(model, cache)
        self.assertEqual(model.rope_deltas.moved_to, 'cuda:1')

    def test_faithact_format_variant_leaves_noun_extraction_unchanged(self):
        class VLM:
            def generate(self,image,prompt,**kwargs):
                return prompt
        adapter = FinalFormatAdapter(VLM(),'Is a car visible?')
        self.assertEqual(adapter.generate(None,'Extract nouns'),'Extract nouns')
        self.assertIn('Yes or No',adapter.generate(object(),'Evidence-conditioned answer'))

    def test_exhausted_point_is_not_given_eighteen_more_repairs(self):
        backend = Scripted([Step('the light is red')]*7,['inconclusive']*4)
        result = solve(backend,None,'q',max_steps=6)
        self.assertEqual(len(backend.contexts),7)
        self.assertEqual(len(result['ledger']),1)
        self.assertEqual(result['status'],'answered_with_uncertainty')
        self.assertEqual(result['trace'][-1]['event'],'redirection_exhausted')

    def test_unverified_hypothesis_is_not_repeated_as_premise(self):
        ledger = [Step('Uncertain whether: the light is red',evidence_question='What color is the light?',
                       verification='uncertain')]
        text = Online.ledger_text(ledger)
        self.assertNotIn('the light is red',text)
        self.assertIn('What color is the light?',text)
        self.assertIn('uncertain',text)

    def test_reworded_claim_cannot_reopen_exhausted_question(self):
        backend = Scripted([Step('the light is red',evidence_question='What color is the light?')]*4
                           + [Step('the signal appears green',evidence_question='WHAT color is the light!')]*3,
                           ['inconclusive']*4)
        result = solve(backend,None,'q')
        self.assertEqual(sum('audit' in e for e in result['trace']),4)
        self.assertEqual(sum(e.get('event') == 'repeat_of_exhausted_question'
                             for e in result['trace']),3)
        self.assertEqual(result['trace'][-1]['event'],'redirection_exhausted')
        self.assertEqual(result['status'],'answered_with_uncertainty')

    def test_repeated_exhausted_question_is_redirected_to_new_evidence(self):
        repeated = Step('red light', evidence_question='Light color?')
        backend = Scripted([repeated]*5 + [Step('the car is stopped', evidence_question='Is the car moving?'),
                                          Step('conditional answer', 'deduction', final_answer='C')],
                           ['inconclusive']*4 + ['supported']*2)
        result = solve(backend, None, 'q')
        self.assertEqual(result['answer'], 'C')
        self.assertEqual(sum('audit' in e for e in result['trace']), 6)
        self.assertIn('Choose a different evidence point', backend.contexts[5][1])

    def test_new_evidence_then_conditional_answer_can_continue(self):
        backend = Scripted([Step('unclear light',evidence_question='What color is the light?')]*4
                           + [Step('the car is stopped',evidence_question='Is the car moving?'),
                              Step('conditional conclusion','deduction',evidence_question='What color is the light?',
                                   final_answer='C')],['inconclusive']*4+['supported']*2)
        result = solve(backend,None,'q')
        self.assertEqual(result['answer'],'C')
        self.assertEqual(len(result['ledger']),3)
        self.assertEqual(result['status'],'answered_with_uncertainty')

    def test_all_repair_question_aliases_remain_exhausted(self):
        backend = Scripted([Step('ambiguous light',evidence_question=q) for q in
                           ['Light color?','Signal color?','Lamp color?','Traffic light color?']]
                           + [Step('independent fact',evidence_question='Road clear?')]
                           + [Step('a reworded light claim',evidence_question='Signal color?')]*3,
                           ['inconclusive']*4+['supported'])
        result = solve(backend,None,'q')
        self.assertEqual(sum('audit' in e for e in result['trace']),5)
        self.assertEqual(result['trace'][-1]['event'],'redirection_exhausted')

    def test_bf16_placement_never_offloads(self):
        sizes = {'visual':5,'model.embed_tokens':3,'lm_head':3,'model.layers.0':2,'model.layers.1':2}
        placement = pack_modules(sizes,{0:8,1:8})
        self.assertNotIn('',placement)
        self.assertTrue(all(isinstance(device,int) for device in placement.values()))
        for device in (0,1):
            self.assertLessEqual(sum(size for name,size in sizes.items() if placement[name]==device),8)
        with self.assertRaises(ValueError):
            pack_modules(sizes,{0:4,1:4})

    def test_bf16_uses_combined_free_memory(self):
        self.assertEqual(select_devices([(0,6000),(1,4000)],17000),[])
        self.assertEqual(select_devices([(0,10000),(1,10000),(2,3000)],17000),[(0,10000),(1,10000)])

    def test_question_instructions_do_not_override_step_or_audit(self):
        q = 'Is a car visible?\nPlease answer directly with a single word or number.'
        self.assertEqual(task_question(q), 'Is a car visible?')
        self.assertIn('Yes or No', answer_format(q))
        self.assertNotIn('Please answer directly', cot_prompt(q))
        self.assertIn('option letter', answer_format('Is it red?\nA. Yes\nB. No'))
        vlmeval = 'Question: What?\nOptions:\nA. one\nB. two\nPlease select the correct answer from the options above.'
        self.assertNotIn('Please select', cot_prompt(vlmeval))
        concise = cot_prompt(vlmeval, 'grounded_concise_v3')
        self.assertNotIn('Please select', concise)
        self.assertIn('1 to 3 short numbered evidence statements', concise)
        preserved = cot_prompt(vlmeval, 'preserve_select_v3')
        self.assertIn('Please select', preserved)
        self.assertIn('FINAL ANSWER:', preserved)

    def test_localization_uses_processed_not_original_pixels(self):
        class VLM:
            def model_image_size(self, image):
                return (100,200)
            def generate(self, image, prompt, **kwargs):
                return '{"box":[25,50,75,150]}'
        self.assertEqual(Online(VLM()).locate(Image.new('RGB',(1000,2000)), 'region'),
                         (.25,.25,.75,.75))

    def test_counterfactual_gate_runs_target_and_controls(self):
        class VLM:
            def __init__(self):
                self.scores = iter([.95, .35, .90, .92])
            def generate(self, *args, **kwargs):
                return '{"box":[8,8,16,16]}'
            def yes_probability(self, *args):
                return next(self.scores)
        image = Image.new('RGB', (40,40), 'white')
        result = Online(VLM(), counterfactual=True).audit(
            image, 'q', Step('visible object', evidence_question='object region', box=(.2,.2,.4,.4)), [])
        self.assertEqual(result.status, 'supported')
        self.assertAlmostEqual(result.scores['specificity'], .56)
        self.assertEqual(len(result.scores['control_scores']), 2)

    def test_counterfactual_global_damage_is_not_specific_evidence(self):
        class VLM:
            def __init__(self):
                self.scores = iter([.95, .35, .30, .32])
            def generate(self, *args, **kwargs):
                return '{"box":[8,8,16,16]}'
            def yes_probability(self, *args):
                return next(self.scores)
        result = Online(VLM(), counterfactual=True).audit(
            Image.new('RGB', (40,40)), 'q', Step('fact', evidence_question='region', box=(.2,.2,.4,.4)), [])
        self.assertEqual(result.status, 'inconclusive')

    def test_repair_does_not_pollute_next_context(self):
        b = Scripted([Step('wrong'), Step('right'), Step('therefore', 'deduction', final_answer='A')],
                     ['contradicted', 'supported', 'supported'])
        result = solve(b, None, 'question')
        self.assertEqual(result['answer'], 'A')
        self.assertEqual(b.contexts[2], (['right'], ''))
        self.assertEqual(result['trace'][2]['parents'], [0])

    def test_no_silent_pass_on_exhaustion(self):
        b = Scripted([Step('ambiguous')]*4, ['inconclusive']*4)
        result = solve(b, None, 'q', max_steps=1)
        self.assertEqual(result['answer'], 'B')
        self.assertEqual(result['ledger'][0]['verification'], 'uncertain')
        self.assertEqual(result['status'], 'answered_with_uncertainty')
        self.assertEqual(len(b.contexts), 4)  # Initial attempt plus three repairs.

    def test_uncertain_point_continues_and_status_is_inherited(self):
        b = Scripted([Step('ambiguous')]*4 + [Step('conditional answer', 'deduction', final_answer='C')],
                     ['inconclusive']*4 + ['supported'])
        result = solve(b, None, 'q', max_steps=2)
        self.assertEqual(result['answer'], 'C')
        self.assertEqual(result['status'], 'answered_with_uncertainty')
        self.assertEqual(len(result['ledger']), 2)
        self.assertTrue(b.contexts[4][0][0].startswith('Uncertain whether:'))
        self.assertEqual(result['ledger'][1]['verification'], 'uncertain')

    def test_third_repair_can_recover(self):
        b = Scripted([Step('bad')]*3 + [Step('fixed'), Step('answer', 'deduction', final_answer='A')],
                     ['contradicted']*3 + ['supported']*2)
        result = solve(b, None, 'q', max_steps=2)
        self.assertEqual(result['status'], 'verified_by_model')
        self.assertEqual(result['ledger'][0]['claim'], 'fixed')

    def test_repair_budget_cannot_exceed_three(self):
        with self.assertRaises(ValueError):
            solve(Scripted([], []), None, 'q', repairs=4)

    def test_first_step_deduction_cannot_bypass_image(self):
        b = Scripted([Step('answer', 'deduction', final_answer='A')], [])
        result = solve(b, None, 'q', repairs=0, max_steps=1)
        self.assertEqual(result['status'], 'answered_with_uncertainty')
        self.assertEqual(result['ledger'][0]['verification'], 'uncertain')

    def test_parse_error_is_bounded(self):
        b = Scripted([ValueError('bad')]*2, [])
        self.assertEqual(solve(b, None, 'q', repairs=1, max_steps=1)['status'], 'answered_with_uncertainty')

    def test_step_limit(self):
        b = Scripted([Step('fact')], ['supported'])
        self.assertEqual(solve(b, None, 'q', max_steps=1)['status'], 'answered_with_uncertainty')

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

    def test_visual_final_is_audited_not_rejected_for_format(self):
        class VLM:
            def yes_probability(self, *args):
                return .95
        audit = Online(VLM()).audit(Image.new('RGB',(20,20)), 'q', Step('fact', final_answer='A'), [])
        self.assertEqual(audit.status, 'supported')
        self.assertEqual(audit.scores['answer_entailment'], .95)

    def test_verified_fact_survives_bad_answer_mapping(self):
        class VLM:
            def __init__(self):
                self.values = iter([.95, .1])
            def yes_probability(self, *args):
                return next(self.values)
        audit = Online(VLM()).audit(Image.new('RGB',(20,20)), 'q', Step('fact', final_answer='A'), [])
        self.assertEqual(audit.status, 'supported')
        self.assertTrue(audit.scores['answer_withheld'])

    def test_option_counterfactual_withholds_nonmaximal_answer(self):
        class VLM:
            def yes_probability(self, *args):
                return .95
            def log_probability(self, image, prompt, completion):
                return {'A':-3., 'B':-1., 'C':-2.}[completion]
        question = 'Question: count?\nA. one\nB. two\nC. three'
        audit = Online(VLM()).audit(
            Image.new('RGB',(20,20)), question, Step('Two objects are visible', final_answer='A'), [])
        self.assertEqual(audit.status, 'supported')
        self.assertEqual(audit.scores['option_counterfactual_best'], 'B')
        self.assertTrue(audit.scores['answer_withheld'])

    def test_withheld_answer_does_not_end_reasoning(self):
        class Withheld(Scripted):
            def audit(self, image, question, step, ledger):
                return Audit('supported',{'answer_withheld':not bool(ledger)})
        backend = Withheld([Step('verified observation',final_answer='A'),
                            Step('correct conclusion','deduction',final_answer='B')],[])
        result = solve(backend,None,'q')
        self.assertEqual(result['answer'],'B')
        self.assertIsNone(result['ledger'][0]['final_answer'])
        self.assertEqual(result['ledger'][0]['verification'],'verified')
        self.assertEqual(len(result['ledger']),2)

    def test_online_uncertain_finalization_preserves_initial_mcq_candidate(self):
        class VLM:
            def generate(self, *args, **kwargs):
                return 'B'
        backend = Online(VLM())
        backend.fallback_answer = 'A'
        uncertain = Step('Uncertain whether: visual estimate', verification='uncertain')
        self.assertEqual(backend.finalize(None, 'Question?\nA. one\nB. two', [uncertain]), 'A')
        verified = Step('A different fact is visible', verification='verified')
        self.assertEqual(backend.finalize(None, 'Question?\nA. one\nB. two', [verified]), 'B')

    def test_answer_extraction(self):
        self.assertEqual(extract_answer('reason\nFINAL ANSWER: B.'), 'B')
        self.assertEqual(extract_answer('Thus, Yes.'), 'Yes')
        self.assertEqual(extract_answer('Thus, the answer is **A**.'), 'A')
        self.assertEqual(extract_answer(None), '')
        self.assertEqual(extract_answer('B. A blue car.', 'A. red B. blue'), 'B')
        self.assertEqual(extract_answer('Thus, the final answer is:\nB. No', 'A. Yes B. No'), 'B')
        self.assertEqual(extract_answer('Reasoning\nFinal answer:\n\nGreen'), 'Green')
        self.assertEqual(extract_answer('Reasoning mentions A and B.\nGiven this:\n\nB', 'A. Yes B. No'), 'B')
        question = ('Question: How many lanes?\nOptions:\nA. 1 lane\nB. 2 lanes\nC. 3 lanes\n'
                    'Please select the correct answer from the options above.')
        self.assertEqual(extract_answer(
            'Given these observations, the correct answer is:\n\nB. 2 lanes\n\nThis follows from the arrow.',
            question), 'B')
        distance = ('Question: distance?\nOptions:\nA. 5.5 meters\nB. 9.5 meters\nC. 13.5 meters')
        self.assertEqual(extract_answer(
            'Thus, the horizontal distance is approximately 5.5 meters.', distance), 'A')
        self.assertEqual(extract_answer(
            'I compare the visible distances.\n\nB. 9.5 meters', distance), 'B')
        self.assertNotEqual(extract_answer(
            'The options are 5.5 meters, 9.5 meters, and 13.5 meters.', distance), 'A')

    def test_empty_and_colon_labeled_answers(self):
        self.assertEqual(extract_answer('  \n'), '')
        self.assertEqual(extract_answer('A: 2 green lights','A. 2 B. 3'), 'A')
        self.assertEqual(extract_answer('A: 2 B: 3','A. 2 B. 3'), 'A: 2 B: 3')
        self.assertEqual(extract_answer('A: a label in open text','Describe the label'), 'A: a label in open text')


if __name__ == '__main__':
    unittest.main()
