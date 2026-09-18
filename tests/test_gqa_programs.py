import json
from pathlib import Path
import tempfile
import unittest
from efa.gqa_programs import compile_episode
from efa.native import training_reads


class NativeProgramTests(unittest.TestCase):
    def fixture(self):
        return dict(width=100,height=100,question='What is the person holding?',answer='cup',
            source_program=[dict(operation='select',argument='person (1)',dependencies=[]),
                dict(operation='relate',argument='_,holding,o (2)',dependencies=[0]),
                dict(operation='query',argument='name',dependencies=[1])],
            native_program_objects={
                '1':dict(name='man',bbox=[0,0,50,90],attributes=[],relations=[dict(object='2',name='holding')]),
                '2':dict(name='cup',bbox=[50,20,80,60],attributes=[],relations=[])})

    def test_object_direction_uses_anchor_edge_and_both_original_regions(self):
        episode=compile_episode(self.fixture())
        relation=episode['steps'][1]['atoms'][0]
        self.assertEqual(relation['native_evidence']['edge'],dict(source='1',target='2',name='holding'))
        self.assertEqual(relation['region']['boxes'],[[0,0,50,90],[50,20,80,60]])
        self.assertEqual(episode['steps'][2]['parents'],[2])
        self.assertNotIn('cup',episode['steps'][2]['atoms'][0]['subject'])
        self.assertEqual(episode['steps'][2]['atoms'][0]['subject'],'object that person is holding')
        self.assertEqual(episode['steps'][2]['atoms'][0]['region']['boxes'],[[50,20,80,60],[0,0,50,90]])
        self.assertIsNone(episode['runtime_audit'])

    def test_relation_direction_is_not_inferred_from_question_answer(self):
        row=self.fixture();row['source_program'][1]['argument']='_,holding,s (2)'
        with self.assertRaisesRegex(ValueError,'missing_native_relation_edge'):
            compile_episode(row)
        row=self.fixture();row['answer']='bottle'
        with self.assertRaisesRegex(ValueError,'final_answer_disagrees'):
            compile_episode(row)

    def test_missing_edge_and_noncausal_dependencies_cannot_be_compiled(self):
        row=self.fixture();row['native_program_objects']['1']['relations']=[]
        with self.assertRaisesRegex(ValueError,'missing_native_relation_edge'):
            compile_episode(row)
        row=self.fixture();row['source_program'][1]['dependencies']=[2]
        with self.assertRaisesRegex(ValueError,'unsupported_non_linear_program'):
            compile_episode(row)

    def test_reader_training_preserves_regions_and_excludes_rejections_and_dev(self):
        row=compile_episode(self.fixture())
        row.update(id='native',split='train',image='original.jpg',benchmark_dedup_status='pending',
                   annotation_status='teacher_reviewed_episode')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[row,dict(row,id='heldout',split='dev'),dict(row,id='rejected',annotation_status='teacher_review_rejected')]
            (root/'teacher_reviewed_episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.assertEqual(training_reads(root,False,'teacher_reviewed_episodes.jsonl'),[])
            reads=training_reads(root,True,'teacher_reviewed_episodes.jsonl')
        self.assertEqual(len(reads),3)
        self.assertEqual(reads[1]['boxes'],[[0,0,50,90],[50,20,80,60]])
        self.assertEqual({r['episode_id'] for r in reads},{'native'})
