from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from efa.clevr_programs import compile_episode,branches
from efa.export import export
from PIL import Image


def fixture():
    def node(function,inputs=[],values=[]):return dict(function=function,inputs=inputs,value_inputs=values)
    program=[node('scene'),node('filter_color',[0],['red']),node('count',[1]),
             node('scene'),node('filter_shape',[3],['cube']),node('count',[4]),node('greater_than',[2,5])]
    question=dict(question_index=9,question_family_index=2,image_filename='CLEVR_train_000000.png',
                  question='Are there more red objects than cubes?',answer='yes',program=program)
    scene=dict(image_filename=question['image_filename'],objects=[dict(color='red',shape='sphere'),
               dict(color='red',shape='cube'),dict(color='blue',shape='sphere')])
    return question,scene


class ClevrProgramTests(unittest.TestCase):
    def test_counts_are_native_observations_and_comparison_uses_only_parents(self):
        question,scene=fixture();row=compile_episode(question,scene,480,320)
        self.assertEqual([s['atoms'][0]['value'] for s in row['steps']],['2','1','yes'])
        self.assertEqual([s['atoms'][0]['kind'] for s in row['steps']],['visual','visual','premise'])
        self.assertEqual(row['steps'][2]['parents'],[1,2])
        self.assertNotIn('region',row['steps'][2]['atoms'][0])
        self.assertEqual(row['steps'][0]['atoms'][0]['region']['boxes'],[[0,0,480,320]])
        self.assertIsNone(row['runtime_audit'])

    def test_reference_disagreement_wrong_image_and_noncausal_program_are_rejected(self):
        question,scene=fixture();question['answer']='no'
        with self.assertRaisesRegex(ValueError,'disagrees'):compile_episode(question,scene,480,320)
        question,scene=fixture();scene['image_filename']='another.png'
        with self.assertRaisesRegex(ValueError,'identity'):compile_episode(question,scene,480,320)
        question,_=fixture();question['program'][1]['inputs']=[1]
        with self.assertRaisesRegex(ValueError,'Noncausal'):branches(question)

    def test_empty_filtered_set_is_zero_from_complete_scene(self):
        question,scene=fixture();question['program'][1]['value_inputs']=['green'];question['answer']='no'
        row=compile_episode(question,scene,480,320)
        self.assertEqual(row['steps'][0]['atoms'][0]['value'],'0')
        self.assertEqual(row['steps'][0]['atoms'][0]['native_evidence']['object_indices'],[])

    def test_export_teaches_deduction_reading_without_images_or_verified_labels(self):
        question,scene=fixture();row=compile_episode(question,scene,48,32)
        row.update(split='train',image='image.png')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);Image.new('RGB',(48,32)).save(root/'image.png')
            (root/'raw_tasks.jsonl').write_text('')
            (root/'native_episodes.jsonl').write_text(json.dumps(row)+'\n')
            export(root,root/'sft',True)
            examples=json.loads((root/'sft/efa_premises.json').read_text())
            self.assertEqual(len(examples),1)
            self.assertEqual(examples[0]['images'],[])
            prompt=examples[0]['messages'][0]['content']
            self.assertNotIn('<image>',prompt)
            self.assertIn('step1.atom0@1',prompt)
            self.assertIn('step2.atom0@1',prompt)
            self.assertNotIn('"verified": true',prompt)
            self.assertEqual(examples[0]['messages'][1]['content'],'yes')
