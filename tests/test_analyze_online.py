import json
from pathlib import Path
import tempfile
import unittest

from trainingfree.analyze_online import analyze, diagnose_row
from trainingfree.answers import PARSER_VERSION


QUESTION = 'Q\nA. left\nB. right'


def audit(answer, attempt=0, scores=None, status='inconclusive'):
    return {'step':0,'attempt':attempt,'candidate':{'claim':'A visible object','final_answer':answer},
            'audit':{'status':status,'scores':scores or {}}}


class AnalyzeOnlineTests(unittest.TestCase):
    def test_missing_initial_answer_and_initial_parse_failure_do_not_inflate_gains(self):
        row = {'id':'0','question':QUESTION,'reference':'B','answer':'B','correct':True,
               'trace':[audit(None),audit('A',1),audit('B',2)]}
        detail = diagnose_row(row)
        self.assertTrue(detail['initial_answer_missing'])
        self.assertIsNone(detail['initial_correct'])
        self.assertFalse(detail['gain'])
        self.assertFalse(detail['paired_initial_final'])
        row['trace'] = [{'step':0,'attempt':0,'parse_error':'bad JSON'},audit('A',1),audit('B',2)]
        detail = diagnose_row(row)
        self.assertTrue(detail['initial_answer_missing'])
        self.assertEqual(detail['initial_attempt'],0)
        self.assertFalse(detail['gain'])
        self.assertEqual(detail['parse_errors'],1)

    def test_mask_readback_support_repairs_and_finalization_counts(self):
        row = {'id':'0','question':QUESTION,'reference':'B','answer':'B','correct':True,
               'status':'answered_with_uncertainty','elapsed_seconds':30,'calls':9,
               'trace':[
                   audit('A',scores={'support':.2}),
                   audit('B',1,{'support':.7,'target_mask':.3,'control_scores':[.6,.65],
                                'repair_readback':{'repair_round':1,'observation':'visible wheel'},
                                'answer_withheld':True,'option_counterfactual_scores':{'A':-2,'B':-1}}),
                   {'step':0,'event':'uncertainty_commit'},
                   {'step':1,'event':'repeat_of_exhausted_question'},
                   {'step':1,'event':'repeat_of_verified_point'},
                   {'event':'redirection_exhausted'}],
               'finalization':{'source':'highest_support_tentative_candidate'}}
        detail = diagnose_row(row)
        self.assertTrue(detail['gain'])
        self.assertEqual((detail['audits'],detail['repair_attempts'],detail['repair_audits']),(2,1,1))
        self.assertEqual((detail['cf_checks'],detail['cf_mask_calls'],detail['cf_control_masks']),(1,3,2))
        self.assertEqual(detail['option_consistency_checks'],1)
        self.assertEqual(detail['logged_readback_n'],1)
        self.assertEqual(detail['answer_withheld'],1)
        self.assertEqual(detail['uncertainty_commits'],1)
        self.assertEqual((detail['repeated_question_events'],detail['repeated_events_total']),(1,2))
        self.assertAlmostEqual(detail['mean_support'],.45)
        self.assertEqual(detail['finalization_source'],'highest_support_tentative_candidate')

    def test_partial_file_aggregate_pair_denominators_and_stored_score(self):
        rows = [
            {'id':'0','reference':'B','answer':'B','correct':True,'status':'verified_by_model',
             'trace':[audit('A'),audit('B',1)],'elapsed_seconds':10,'calls':2},
            {'id':'1','reference':'B','answer':'A','correct':False,'status':'answered_with_uncertainty',
             'trace':[audit('B')],'elapsed_seconds':20,'calls':4},
            {'id':'2','reference':'B','raw':None,'answer':'B','correct':True,
             'status':'answered_with_uncertainty','trace':[audit(None)]},
            {'id':'3','reference':'B','raw':'','answer':'B','correct':False,'status':'error','trace':[]}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'online.jsonl'
            cases = Path(folder)/'cases.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
            cases.write_text(''.join(json.dumps({'id':str(i),'question':QUESTION,'reference':'B'})+'\n'
                                     for i in range(10)),encoding='utf-8')
            result = analyze(path,cases)
            summary = result['summary']
            self.assertEqual(summary['parser_version'],PARSER_VERSION)
            self.assertEqual((summary['completed_n'],summary['correct'],summary['error_n']),(4,2,1))
            self.assertEqual((summary['uncertain_n'],summary['verified_n']),(2,1))
            self.assertEqual((summary['initial_answer_n'],summary['initial_missing_n'],summary['paired_n']),(2,2,2))
            self.assertEqual(summary['initial_accuracy'],.5)
            self.assertEqual(summary['gains_ids'],['0'])
            self.assertEqual(summary['losses_ids'],['1'])
            self.assertEqual(summary['mean_latency_seconds'],15)
            self.assertEqual(summary['mean_calls'],3)
            self.assertEqual(result['details'][2]['final_answer'],'B')
            self.assertFalse(result['details'][3]['final_rescored_correct'])
            rows[0]['answer'] = 'A'
            rows[0]['question'] = QUESTION
            detail = diagnose_row(rows[0])
            self.assertTrue(detail['final_correct'])
            self.assertTrue(detail['stored_rescore_disagreement'])
            rescored = diagnose_row(rows[0],{'question':QUESTION,'reference':'B'})
            self.assertFalse(rescored['final_correct'])
            self.assertEqual(rescored['final_score_source'],'reference_rescore')
            self.assertTrue(rescored['stored_correct'])

    def test_cases_rescore_empty_raw_instead_of_stale_true_and_preserve_diagnostic(self):
        row = {'id':'0','reference':'B','raw':'','answer':'B','correct':True,
               'status':'answered_with_uncertainty','trace':[audit(None),audit('A',1)]}
        with tempfile.TemporaryDirectory() as folder:
            path,cases = Path(folder)/'online.jsonl',Path(folder)/'cases.jsonl'
            path.write_text(json.dumps(row)+'\n',encoding='utf-8')
            cases.write_text(json.dumps({'id':'0','question':QUESTION,'reference':'B'})+'\n',encoding='utf-8')
            with_cases = analyze(path,cases)
            without_cases = analyze(path)
            self.assertEqual(with_cases['summary']['correct'],0)
            self.assertEqual(with_cases['summary']['stored_score_diagnostic'],{'n':1,'correct':1,'accuracy':1})
            self.assertEqual(with_cases['details'][0]['answer_source'],'raw')
            self.assertEqual(without_cases['summary']['correct'],1)
            self.assertEqual(without_cases['details'][0]['final_score_source'],'stored_correct')
            self.assertEqual(with_cases['summary']['gains_ids'],[])
            self.assertEqual(without_cases['summary']['gains_ids'],[])

    def test_option_consistency_is_not_a_mask_and_reference_mismatch_fails(self):
        row = {'id':'0','question':QUESTION,'reference':'B','answer':'B','correct':True,
               'trace':[audit('A',scores={'option_counterfactual_scores':{'A':-2,'B':-1}})]}
        detail = diagnose_row(row)
        self.assertEqual(detail['option_consistency_checks'],1)
        self.assertEqual(detail['cf_mask_calls'],0)
        with self.assertRaisesRegex(ValueError,'Reference mismatch'):
            diagnose_row(row,{'question':QUESTION,'reference':'A'})

    def test_no_public_question_cannot_create_gain_from_unvalidated_initial_option(self):
        detail = diagnose_row({'id':'0','reference':'B','answer':'B','correct':True,
                               'trace':[audit('N')]})
        self.assertTrue(detail['final_correct'])
        self.assertIsNone(detail['initial_correct'])
        self.assertEqual(detail['initial_unavailable_reason'],'public_question_unavailable')
        self.assertFalse(detail['gain'])


if __name__ == '__main__':
    unittest.main()
