"""Original CLEVR counting observations followed by explicit program deductions."""
import argparse
from collections import Counter
import hashlib
import heapq
import io
import json
from pathlib import Path
import zipfile
from PIL import Image
from efa.data import partition
from efa.remote_zip import RangeReader

REPO='dali-does/clevr-math'
REVISION='6a30110f887edd7edbad033275aa853ddd8c4a26'
ARCHIVE='data/CLEVR_v1.0.zip'
COMPARISONS={'equal_integer':'is the first count equal to the second count',
             'less_than':'is the first count less than the second count',
             'greater_than':'is the first count greater than the second count'}


def count_spec(program,index):
    if type(index) is not int or not 0<=index<len(program):raise ValueError('Invalid program input')
    root=program[index]
    if root['function']!='count' or len(root['inputs'])!=1:raise ValueError('Expected a count branch')
    previous=index;index=root['inputs'][0];filters={}
    while True:
        if type(index) is not int or not 0<=index<previous:raise ValueError('Noncausal program input')
        node=program[index];function=node['function']
        if function=='scene':
            if node['inputs'] or node.get('value_inputs'):raise ValueError('Invalid scene node')
            return filters
        if function not in ['filter_color','filter_shape','filter_material','filter_size']:
            raise ValueError('Unsupported count branch')
        if len(node['inputs'])!=1 or len(node['value_inputs'])!=1:raise ValueError('Invalid filter')
        field=function.removeprefix('filter_');value=node['value_inputs'][0]
        if field in filters and filters[field]!=value:raise ValueError('Contradictory filter description')
        filters[field]=value;previous=index;index=node['inputs'][0]


def branches(question):
    program=question['program'];final=program[-1]
    if final['function'] not in COMPARISONS or len(final['inputs'])!=2:raise ValueError('Unsupported terminal comparison')
    if any(type(i) is not int or not 0<=i<len(program)-1 for i in final['inputs']):raise ValueError('Invalid comparison inputs')
    specs=[count_spec(program,i) for i in final['inputs']]
    if specs[0]==specs[1]:raise ValueError('Identical groups give a trivial comparison')
    return specs


def compile_episode(question,scene,width,height):
    if question['image_filename']!=scene['image_filename']:raise ValueError('Scene/image identity mismatch')
    specs=branches(question);steps=[];counts=[];whole=[0,0,width,height]
    for number,spec in enumerate(specs,1):
        members=[i for i,obj in enumerate(scene['objects']) if all(obj[key]==value for key,value in spec.items())]
        counts.append(len(members))
        subject='all objects'+(' with '+' and '.join(key+' '+value for key,value in sorted(spec.items())) if spec else '')
        atom=dict(subject=subject,predicate='number of matching objects in the complete scene',
            value=str(len(members)),kind='visual',family='count',
            region=dict(boxes=[whole],context_boxes=[whole],insufficient=False,coordinate_system='original_pixels',
                label_basis='complete_native_CLEVR_scene',source_scope='full_scene_required_for_exhaustive_count'),
            native_evidence=dict(filters=spec,object_indices=members,program_node=question['program'][-1]['inputs'][number-1]))
        steps.append(dict(id=number,parents=[],atoms=[atom],text=f'Step {number}: {subject}: {len(members)}'))
    operation=question['program'][-1]['function'];a,b=counts
    truth={'equal_integer':a==b,'less_than':a<b,'greater_than':a>b}[operation]
    answer='yes' if truth else 'no'
    if answer!=question['answer']:raise ValueError('Executed program disagrees with native answer')
    atom=dict(subject='the counts recorded in Step 1 and Step 2',predicate=COMPARISONS[operation],
        value=answer,kind='premise',family='deduction',
        native_evidence=dict(operation=operation,operands=counts,parent_steps=[1,2]))
    steps.append(dict(id=3,parents=[1,2],atoms=[atom],text=f'Step 3: {COMPARISONS[operation]}: {answer}'))
    return dict(id='clevr-count-'+str(question['question_index']),question=question['question'],answer=answer,
        width=width,height=height,steps=steps,source_program=question['program'],native_scene=scene,
        source_question_index=question['question_index'],source_template_family=question['question_family_index'],
        source_split='train',source_image_name=question['image_filename'],
        source_repo=REPO,source_revision=REVISION,source_archive=ARCHIVE,
        annotation_status='native_program_episode',label_basis='executed_original_CLEVR_scene_and_program',
        benchmark_dedup_status='pending',runtime_audit=None)


def build(output,limit=256):
    import ijson
    output=Path(output);identity=dict(repo=REPO,revision=REVISION,archive=ARCHIVE,limit=limit,sampling='image_hash_seed_2027')
    if limit<1:raise ValueError('Positive example limit required')
    if output.exists():
        if json.loads((output/'experiment.json').read_text())!=identity:raise ValueError('Existing experiment differs')
        if (output/'manifest.json').exists() and json.loads((output/'manifest.json').read_text()).get('construction_complete'):
            raise ValueError('Completed version is immutable')
    else:
        output.mkdir(parents=True);(output/'images').mkdir()
        (output/'experiment.json').write_text(json.dumps(identity,indent=2))
    questions_path=output/'source_questions.json';scenes_path=output/'source_scenes.json'
    url=f'https://hf-mirror.com/datasets/{REPO}/resolve/{REVISION}/{ARCHIVE}'
    with RangeReader(url) as reader,zipfile.ZipFile(reader) as archive:
        members={Path(x.filename).name:x for x in archive.infolist() if not x.is_dir()}
        if not questions_path.exists():
            heap=[];seen=set();supported=0
            with archive.open(members['CLEVR_train_questions.json']) as stream:
                for question in ijson.items(stream,'questions.item',use_float=True):
                    if question['split']!='train':continue
                    try:branches(question)
                    except ValueError:continue
                    supported+=1;name=question['image_filename']
                    if name in seen:continue
                    seen.add(name);priority=int(hashlib.sha256(('2027|'+name).encode()).hexdigest(),16)
                    if len(heap)<limit:heapq.heappush(heap,(-priority,name,question))
                    elif priority<-heap[0][0]:heapq.heapreplace(heap,(-priority,name,question))
            questions=sorted((entry[2] for entry in heap),key=lambda q:q['image_filename'])
            questions_path.write_text(json.dumps(questions))
            print(json.dumps(dict(stage='questions_selected',count=len(questions),eligible_programs=supported,range_bytes=reader.bytes_read)),flush=True)
        else:questions=json.loads(questions_path.read_text())
        if len(questions)!=limit:raise ValueError('Insufficient supported programs')
        if not scenes_path.exists():
            wanted={q['image_filename'] for q in questions};scenes={}
            with archive.open(members['CLEVR_train_scenes.json']) as stream:
                for scene in ijson.items(stream,'scenes.item',use_float=True):
                    if scene['image_filename'] in wanted:scenes[scene['image_filename']]=scene
            if len(scenes)!=limit:raise ValueError('Missing original scenes')
            scenes_path.write_text(json.dumps(scenes))
            print(json.dumps(dict(stage='scenes_selected',count=len(scenes),range_bytes=reader.bytes_read)),flush=True)
        else:scenes=json.loads(scenes_path.read_text())
        records=[]
        for index,question in enumerate(questions):
            name=question['image_filename'];path=output/'images'/name
            if not path.exists():path.write_bytes(archive.read(members[name]))
            image=Image.open(path).convert('RGB')
            row=compile_episode(question,scenes[name],*image.size)
            digest=hashlib.sha256(image.tobytes()+str(image.size).encode()).hexdigest()
            row.update(image='images/'+name,image_sha256=digest,split=partition(digest))
            records.append(row)
            if (index+1)%32==0:print(json.dumps(dict(stage='original_images',count=index+1,range_bytes=reader.bytes_read)),flush=True)
        (output/'native_episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        (output/'raw_tasks.jsonl').write_text(''.join(json.dumps({k:v for k,v in r.items() if k!='steps'})+'\n' for r in records))
        manifest=dict(kind='native_count_and_deduction_episodes',count=len(records),steps=3*len(records),
            visual_atoms=2*len(records),premise_atoms=len(records),images=len(records),
            splits=dict(Counter(r['split'] for r in records)),answers=dict(Counter(r['answer'] for r in records)),
            source_repo=REPO,source_revision=REVISION,source_archive=ARCHIVE,
            archive_members={name:dict(crc32=members[name].CRC,uncompressed_bytes=members[name].file_size)
                for name in ['CLEVR_train_questions.json','CLEVR_train_scenes.json']},
            source_questions_sha256=hashlib.sha256(questions_path.read_bytes()).hexdigest(),
            source_scenes_sha256=hashlib.sha256(scenes_path.read_bytes()).hexdigest(),
            archive_size=reader.size,range_bytes_this_run=reader.bytes_read,construction_complete=True,
            label_basis='complete_native_scene_program_execution',independent_teacher_review=False,
            runtime_verified_episodes=0,benchmark_dedup_status='pending',image_edits=False,
            source_view_policy='exhaustive_counts_use_original_full_scene; deduction_uses_two_explicit_counts',
            license='CLEVR_v1.0 LICENSE.txt retained; native rendered images are unmodified',
            limitation='Simple filtered count comparisons; no claim of general real-image coverage')
        (output/'LICENSE.txt').write_bytes(archive.read(members['LICENSE.txt']))
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=256)
    a=p.parse_args();print(json.dumps(build(a.output,a.limit)))
