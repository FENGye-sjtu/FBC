import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from efa.evaluate_episodes import EPISODE_TIERS, reference_components, select_episodes, verification_counts
from efa.format import premise_prompt, read_prompt
from efa.qwen import QwenBackend


class EpisodeSelectionTests(unittest.TestCase):
    def write(self,root,name,rows):
        (root/name).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')

    def row(self,name,status='teacher_reviewed_episode',split='dev',**kwargs):
        return dict(id=name,image_sha256=name,annotation_status=status,split=split,
                    benchmark_dedup_status='cleared',**kwargs)

    def test_explicit_tiers_and_same_hash_case_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for file,status in EPISODE_TIERS.items():
                rows=[self.row(name,status) for name in ('one','two','three')]
                rows.extend([self.row('rejected','teacher_source_rejected'),self.row('train',status,'train'),
                             self.row('calibration',status,'calibration')])
                self.write(root,file,rows)
                selected=select_episodes(root,file,'dev',2,True)
                expected=sorted(rows[:3],key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:2]
                self.assertEqual(selected,expected)
                self.assertEqual(select_episodes(root,file,'calibration',2,True)[0]['id'],'calibration')
                if status!='reviewed_episode':
                    with self.assertRaisesRegex(ValueError,'engineering-only'):
                        select_episodes(root,file,'dev',2,False)
                else:self.assertEqual(select_episodes(root,file,'dev',2,False),expected)

    def test_rejected_train_or_raw_train_still_excludes_heldout_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);file='teacher_reviewed_episodes.jsonl'
            heldout=self.row('shared')
            rejected=self.row('training','teacher_source_rejected','train');rejected['image_sha256']='shared'
            self.write(root,file,[heldout,rejected])
            with self.assertRaisesRegex(ValueError,'Train image leakage'):
                select_episodes(root,file,'dev',1,True)
            self.write(root,file,[heldout]);self.write(root,'raw_tasks.jsonl',[rejected])
            with self.assertRaisesRegex(ValueError,'Train image leakage'):
                select_episodes(root,file,'dev',1,True)

    def test_wrong_tier_and_noncleared_formal_evaluation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);file='reviewed_episodes.jsonl'
            self.write(root,file,[self.row('teacher')])
            with self.assertRaisesRegex(ValueError,'No eligible'):
                select_episodes(root,file,'dev',1,True)
            row=self.row('reviewed','reviewed_episode');row['benchmark_dedup_status']='pending'
            self.write(root,file,[row])
            with self.assertRaisesRegex(ValueError,'Benchmark clearance'):
                select_episodes(root,file,'dev',1,False)
            self.assertEqual(select_episodes(root,file,'dev',1,True),[row])
            with self.assertRaisesRegex(ValueError,'Held-out split'):
                select_episodes(root,file,'train',1,True)


class ReferenceComponentTests(unittest.TestCase):
    def fixture(self):
        def atom(subject,value):
            return dict(subject=subject,predicate='count',value=value,family='count',kind='visual',
                        region=dict(boxes=[[0,0,20,20]]),native_evidence=dict(object_indices=[]))
        return dict(image='source.png',image_sha256='digest',steps=[
            dict(id=1,parents=[],atoms=[atom('yellow things','1')]),
            dict(id=2,parents=[],atoms=[atom('green things','0')]),
            dict(id=3,parents=[1,2],atoms=[dict(subject='counts in Step 1 and Step 2',
                predicate='first greater than second',value='yes',family='deduction',kind='premise')])])

    def test_deduction_has_exact_reference_context_no_crop_or_fake_verification(self):
        class Reader:
            prompt=QwenBackend.prompt
            premise_prompt=QwenBackend.premise_prompt
            propose=QwenBackend.propose

            def __init__(self):self.inputs=[]

            def set_image(self,path):
                self.image_digest='digest';self.premises=[];self.audit_kind='visual';self.calls=[]

            def generate(self,prompt,boxes,max_tokens=32,k=1):
                self.inputs.append((prompt,tuple(boxes)))
                self.calls.append(dict(operation='generate',images=len(boxes)))
                # Deliberately wrong visual predictions: deduction must be
                # labelled reference-conditioned, not receive these predictions.
                return ['yes' if self.audit_kind=='premise' else '9']

        backend=Reader();row=self.fixture()
        with patch.object(QwenBackend,'generate',new=Reader.generate):
            visual,premises=reference_components(backend,Path('.'),row)
        self.assertEqual(len(visual),2);self.assertEqual(len(premises),1)
        self.assertTrue(all(not r['compact_correct'] for r in visual))
        result=premises[0];self.assertTrue(result['correct'])
        self.assertEqual(result['supplied_images'],0);self.assertFalse(result['measured_audit'])
        context=result['exposed_premises']
        self.assertEqual([p['claim']['value'] for p in context],['1','0'])
        self.assertEqual([p['verified'] for p in context],[None,None])
        self.assertEqual(context[1]['claim']['parents'],['step1.atom0@1'])
        self.assertEqual(result['inherited_reference_sources'][1]['annotated_parent_steps'],[])
        self.assertEqual(result['inherited_reference_sources'][0]['region'],row['steps'][0]['atoms'][0]['region'])
        query={k:row['steps'][2]['atoms'][0][k] for k in ('subject','predicate')}
        self.assertEqual(backend.inputs[-1],(premise_prompt(query,context),()))
        self.assertNotIn('yes',backend.inputs[-1][0])
        self.assertEqual(result['calls'],[dict(operation='generate',images=0)])
        first=row['steps'][0]['atoms'][0]
        self.assertEqual(backend.inputs[0][0],read_prompt({k:first[k] for k in ('subject','predicate')},[[0,0,20,20]]))

    def test_native_crop_audit_does_not_invoke_compact_reader(self):
        class Reader:
            prompt=QwenBackend.prompt
            premise_prompt=QwenBackend.premise_prompt
            propose=QwenBackend.propose

            def set_image(self,path):
                self.image_digest='digest';self.premises=[];self.audit_kind='visual';self.calls=[]

            def generate(self,prompt,boxes,max_tokens=32,k=1):
                self.calls.append(dict(operation='native_generate',images=len(boxes)))
                return ['1' if len(boxes) else 'yes']

        backend=Reader();row=self.fixture()
        with patch.object(QwenBackend,'generate',new=Reader.generate):
            visual,premises=reference_components(backend,Path('.'),row,'native_crop')
        self.assertEqual([r['audit'] for r in visual],['1','1'])
        self.assertTrue(all(r['compact'] is None for r in visual))
        self.assertTrue(all(r['compact_correct'] is None for r in visual))
        self.assertEqual([r['audit_correct'] for r in visual],[True,False])
        self.assertEqual(len(premises),1)

    def test_verified_wrong_is_not_hidden_by_unverified_correct(self):
        records=[dict(efa=dict(final=dict(verified=verified),correct=correct))
                 for verified,correct in ((True,True),(True,False),(False,True),(False,False))]
        self.assertEqual(verification_counts(records),dict(verified=2,verified_correct=1,verified_wrong=1))


if __name__=='__main__':unittest.main()
