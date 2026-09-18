import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from efa.export import export
from efa.format import read_prompt, route_prompt


class ExportTests(unittest.TestCase):
    def test_reviewed_multisource_read_matches_runtime_and_excludes_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'source';root.mkdir()
            Image.new('RGB',(100,80),'red').save(root/'image.png')
            row=dict(id='example',split='train',image='image.png',width=100,height=80,
                question='How many bags?',answer='2',benchmark_dedup_status='pending',
                annotation_status='teacher_reviewed_episode',steps=[dict(id=1,parents=[],
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
