import json
from pathlib import Path
import tempfile
import unittest

from trainingfree.analyze_online import analyze
from trainingfree.answers import PARSER_VERSION
from trainingfree.compare import compare
from trainingfree.report import summarize
from trainingfree.rescore import rescore


def initial(answer,attempt=0):
    return {'step':0,'attempt':attempt,'candidate':{'final_answer':answer},
            'audit':{'status':'inconclusive','scores':{}}}


class ReportRescoreTests(unittest.TestCase):
    def test_all_reports_score_the_same_source_and_do_not_count_missing_initial_gains(self):
        rows = [
            {'id':'0','raw':None,'answer':'B','correct':False,'trace':[initial(None),initial('A',1)]},
            {'id':'1','raw':'','answer':'B','correct':True,'trace':[initial('B')]},
            {'id':'2','raw':'The correct answer is option B because the object is visible.',
             'answer':'A','correct':False,'trace':[initial('null')]},
            {'id':'3','answer':'B','correct':False,'trace':[initial('N')]},
            {'id':'4','raw':'FINAL ANSWER: A','answer':'B','correct':True,'trace':[initial('A')]}]
        rows = [dict(row,reference='B',status='answered_with_uncertainty',elapsed_seconds=1,calls=2)
                for row in rows]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, dataset = root/'online.jsonl',root/'cases.jsonl'
            raw = ''.join(json.dumps(row)+'\n' for row in rows)
            path.write_text(raw,encoding='utf-8')
            dataset.write_text(''.join(json.dumps({'id':str(i),'question':'Q\nA. left\nB. right',
                                                  'reference':'B'})+'\n' for i in range(len(rows))),encoding='utf-8')
            (root/'manifest.json').write_text(json.dumps({'methods':'online','limit':5}),encoding='utf-8')
            compared = compare(dataset,[('online',path)])
            analyzed = analyze(path,dataset)
            reported = summarize(root,dataset)
            rescored = rescore(root,dataset)
            self.assertEqual(compared['methods']['online']['correct'],3)
            self.assertEqual(analyzed['summary']['correct'],3)
            self.assertEqual(reported['methods']['online']['correct'],3)
            self.assertEqual(rescored['methods']['online']['correct'],3)
            for report in (compared,analyzed['summary'],reported,rescored):
                self.assertEqual(report['parser_version'],PARSER_VERSION)
            self.assertEqual(analyzed['summary']['stored_score_diagnostic']['correct'],2)
            self.assertEqual(analyzed['summary']['gains_ids'],[])
            self.assertEqual(analyzed['summary']['losses_ids'],['1'])
            transitions = reported['methods']['online']['initial_proposed_answer_to_final']
            self.assertEqual(transitions,{'correct_to_wrong':1,'wrong_to_wrong':1})
            self.assertEqual(reported['methods']['online']['initial_missing_answer_n'],3)
            self.assertEqual([case['answer_source'] for case in rescored['methods']['online']['cases']],
                             ['answer','raw','raw','answer','raw'])
            self.assertEqual(path.read_text(encoding='utf-8'),raw)


if __name__ == '__main__':
    unittest.main()
