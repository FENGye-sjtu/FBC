from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from efa.discrimination import category_negatives,build
from efa.reader_objectives import load_objectives
from PIL import Image


def fixture():
    atom=dict(kind='visual',subject='object beside the bench',predicate='category',value='cat',
              native_evidence=dict(field='name',object_ids=['7']),
              region=dict(boxes=[[0,0,8,8]],context_boxes=[[0,0,10,10]],insufficient=False))
    row=dict(id='episode',image='images/source.png',image_sha256='fixture',split='train',
             annotation_status='teacher_reviewed_episode',benchmark_dedup_status='pending',
             native_program_objects={'7':dict(name='cat',bbox=[0,0,8,8])},steps=[dict(id=1,atoms=[atom])])
    return row,atom


class DiscriminationTests(unittest.TestCase):
    def test_negatives_require_explicit_native_disjoint_categories(self):
        row,atom=fixture();negatives=category_negatives(row,atom)
        self.assertEqual(len(negatives),2)
        self.assertTrue(all(n['value']!='cat' and n['exclusive_group']=='animal_species' for n in negatives))
        row['native_program_objects']['7']['name']='dog'
        self.assertEqual(category_negatives(row,atom),[])
        for ambiguous in ['driver','man','taxi','material']:
            atom['value']=ambiguous;row['native_program_objects']['7']['name']=ambiguous
            self.assertEqual(category_negatives(row,atom),[])

    def test_heldout_and_rejected_rows_never_supply_training_variants(self):
        row,_=fixture();dev=deepcopy(row);dev.update(id='heldout',split='dev')
        rejected=deepcopy(row);rejected.update(id='rejected',annotation_status='teacher_review_rejected')
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';(source/'images').mkdir(parents=True)
            Image.new('RGB',(10,10)).save(source/row['image'])
            (source/'teacher_reviewed_episodes.jsonl').write_text('\n'.join(map(json.dumps,[row,dev,rejected])))
            out=Path(tmp)/'output';manifest=build(source,out)
            self.assertEqual((manifest['count'],manifest['episodes'],manifest['negative_strings']),(1,1,2))
            rows=load_objectives(out,True)
            self.assertEqual(rows[0]['episode_id'],'episode')
            with self.assertRaisesRegex(ValueError,'clearance'):load_objectives(out)
            rows[0]['negatives'][0]['value']='cat'
            (out/'reader_objectives.jsonl').write_text(json.dumps(rows[0])+'\n')
            with self.assertRaisesRegex(ValueError,'exclusivity'):load_objectives(out,True)
