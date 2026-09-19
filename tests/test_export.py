import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from efa.export import export
from efa.format import read_prompt, route_prompt
from efa.format import binding_prompt
from efa.gqa_programs import compile_episode
from tests import test_gqa_programs as gqa_fixtures


class ExportTests(unittest.TestCase):
    def test_reviewed_multisource_read_matches_runtime_and_excludes_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'source';root.mkdir()
            Image.new('RGB',(100,80),'red').save(root/'image.png')
            row=dict(id='example',split='train',image='image.png',width=100,height=80,
                question='How many bags?',answer='2',benchmark_dedup_status='pending',
                annotation_status='teacher_reviewed_episode',steps=[dict(id=1,parents=[],
                text='Step 1: There are 2 bags. <STEP_END>',
                atoms=[dict(subject='bags',predicate='count',value='2',kind='visual',family='count',
                region=dict(boxes=[[0,0,20,30],[40,30,70,60]],context_boxes=[[0,0,100,80]],insufficient=False))])])
            (root/'raw_tasks.jsonl').write_text(json.dumps(row)+'\n')
            heldout=dict(row,id='holdout',split='dev')
            (root/'teacher_reviewed_episodes.jsonl').write_text(json.dumps(row)+'\n'+json.dumps(heldout)+'\n')
            out=Path(directory)/'export';export(root,out,True)
            reads=json.loads((out/'efa_reads.json').read_text())
            routes=json.loads((out/'efa_regions.json').read_text())
            self.assertEqual(len(reads),1)
            self.assertEqual(len(reads[0]['images']),2)
            query=dict(subject='bags',predicate='count')
            self.assertEqual(reads[0]['messages'][0]['content'],
                '<image>\n<image>\n'+read_prompt(query,[[0,0,20,30],[40,30,70,60]]))
            self.assertEqual(routes[0]['messages'][0]['content'],'<image>\n'+route_prompt(query,100,80))
            with Image.open(reads[0]['images'][1]) as crop:
                self.assertEqual(crop.size,(30,30))
            self.assertEqual(json.loads((out/'efa_bindings.json').read_text()),[])
            trajectories=json.loads((out/'efa_trajectories.json').read_text())
            self.assertEqual(len(trajectories),1)
            self.assertEqual([m['role'] for m in trajectories[0]['messages']],['user','assistant'])
            self.assertEqual(trajectories[0]['messages'][0]['content'].count('<image>'),1)
            target=trajectories[0]['messages'][1]['content']
            payload=json.loads(target.removesuffix('<STEP_END>'))
            self.assertEqual((payload['step'],payload['text'],payload['final']),(1,'There are 2 bags.',True))
            bundle=json.loads((out/'efa_episode_bundles.jsonl').read_text())
            self.assertEqual((bundle['id'],len(bundle['steps']),bundle['runtime_audit']),('example',1,None))
            self.assertEqual(bundle['steps'][0]['atoms'][0]['boxes'],[[0,0,20,30],[40,30,70,60]])

    def test_native_binding_uses_ordered_crops_and_neutral_subject_not_answer(self):
        row=compile_episode(gqa_fixtures.NativeProgramTests().fixture())
        row.update(id='binding',split='train',image='image.png',annotation_status='teacher_reviewed_episode')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);Image.new('RGB',(100,100)).save(root/'image.png')
            (root/'raw_tasks.jsonl').write_text('')
            (root/'teacher_reviewed_episodes.jsonl').write_text(json.dumps(row)+'\n'+json.dumps(dict(row,split='dev'))+'\n')
            export(root,root/'sft',True)
            bindings=json.loads((root/'sft/efa_bindings.json').read_text())
            self.assertEqual(len(bindings),3)
            final=bindings[-1]
            boxes=[[50,20,80,60],[0,0,50,90],[0,0,100,100]]
            self.assertEqual(final['messages'][0]['content'],'<image>\n'*3+binding_prompt(boxes))
            self.assertEqual(final['messages'][1]['content'],'object that person is holding')
            self.assertNotIn('cup',json.dumps(final['messages']))
            with Image.open(final['images'][0]) as crop:self.assertEqual(crop.size,(30,40))
