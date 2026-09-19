#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=7000 {print $1,$2}' | sort -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then echo 'WAIT_FOR_7000_MIB_FREE'; exit 3; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -i "$gpu" --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader
python - <<'PY'
from collections import Counter
import copy
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
from PIL import Image
import yaml
from efa.export import export
from scripts.package_adjudication import LABEL_BASIS,validate_episode

root=Path('/data_cinema/zqc/VBC')
adapter=root/'runs/mixed_program_sft_2b_v0'
output=root/'runs/codex_sft_2b_v0'
dataset=root/'data/derived/codex_sft_2b_v0'
config_path=root/'runs/codex_sft_2b_v0.yaml'
for path in (output,dataset,config_path):
    if path.exists():raise ValueError('Preserve existing artifact: '+str(path))
if version('llamafactory')!='0.9.3' or version('transformers')!='4.51.3':
    raise ValueError('Recheck LoRA continuation and epoch accounting before changing the pinned framework versions')
for name in ['adapter_model.safetensors','adapter_config.json','trainer_state.json','all_results.json','precision_policy.json']:
    path=adapter/name
    if not path.is_file() or not path.stat().st_size:raise ValueError('Missing completed SFT artifact: '+name)
state=json.loads((adapter/'trainer_state.json').read_text())
if state.get('global_step')!=2400 or state.get('max_steps')!=2400:
    raise ValueError('Parent SFT must have completed 2400 updates')
result=json.loads((adapter/'all_results.json').read_text())
if not math.isfinite(result.get('train_loss',float('nan'))) or result.get('train_runtime',0)<=0:
    raise ValueError('Parent SFT completion metrics are required')
if json.loads((adapter/'precision_policy.json').read_text()).get('mode')!='bf16_vision_nf4_language':
    raise ValueError('Parent SFT precision must be BF16 vision and NF4 language')
lora=json.loads((adapter/'adapter_config.json').read_text())
if lora.get('r')!=16 or set(lora.get('target_modules',[]))!={'q_proj','k_proj','v_proj','o_proj'}:
    raise ValueError('Unexpected parent LoRA rank or target modules')
parent_sha=hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()

sources=[root/'data/derived/codex_adjudicated_v2',root/'data/derived/codex_curated45_adjudicated_v1']
selected=[];components=[];seen_ids=set();seen_train_images=set();train_images=set();heldout_images=set()
for source in sources:
    source=source.resolve()
    manifest_path=source/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if not manifest.get('construction_complete') or manifest.get('label_basis')!=LABEL_BASIS:
        raise ValueError('A completed Codex visual-adjudication source is required')
    raw_path=source/'raw_tasks.jsonl'
    raw=[json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    registry={row['id']:(row['split'],row['image_sha256']) for row in raw}
    if len(registry)!=len(raw):raise ValueError('Duplicate source registry ID')
    for split,digest in registry.values():
        (train_images if split=='train' else heldout_images).add(digest)
    file=source/'reviewed_episodes.jsonl'
    episodes=[json.loads(line) for line in file.read_text().splitlines() if line.strip()]
    if len({row['id'] for row in episodes})!=len(episodes):raise ValueError('Duplicate reviewed episode ID')
    if dict(Counter(row['split'] for row in episodes))!=manifest['accepted_split_counts']:
        raise ValueError('Reviewed split counts differ from the frozen source manifest')
    train=[]
    for row in episodes:
        if row.get('annotation_status')!='reviewed_episode' or row.get('label_basis')!=LABEL_BASIS:
            raise ValueError('Only explicitly Codex-reviewed episodes may enter this run')
        if registry.get(row['id'])!=(row['split'],row['image_sha256']):
            raise ValueError('Reviewed episode differs from the source split/image registry')
        if row['split']!='train':continue
        if row['id'] in seen_ids or row['image_sha256'] in seen_train_images:
            raise ValueError('Repeated training episode or image across the selected sources')
        seen_ids.add(row['id']);seen_train_images.add(row['image_sha256'])
        validate_episode(row)
        original=(source/row['image']).resolve()
        if not original.is_relative_to(source):raise ValueError('Original image escapes its approved source')
        with Image.open(original) as image:
            image=image.convert('RGB')
            digest=hashlib.sha256(image.tobytes()+str(image.size).encode()).hexdigest()
        if digest!=row['image_sha256']:raise ValueError('Reviewed image pixels changed')
        # The view contains no raw QA and no native/teacher episode file. Absolute
        # original-image paths survive export without copying or altering images.
        prepared=copy.deepcopy(row);prepared['image']=str(original);train.append(prepared)
    if len(train)!=manifest['accepted_split_counts'].get('train',0) or not train:
        raise ValueError('Missing reviewed training episodes')
    selected.append(train)
    components.append(dict(source=str(source),manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        episode_file='reviewed_episodes.jsonl',episode_file_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        registry_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        train_episodes=len(train),heldout_episodes=len(episodes)-len(train),label_basis=LABEL_BASIS))
if train_images&heldout_images:raise ValueError('Train/held-out image collision across the source registries')

dataset.mkdir(parents=True)
combined={};descriptor={}
for index,rows in enumerate(selected):
    view=dataset/'reviewed_inputs'/f'source_{index}';view.mkdir(parents=True)
    (view/'raw_tasks.jsonl').write_text('',encoding='utf-8')
    (view/'reviewed_episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    destination=dataset/f'source_{index}'
    # The engineering flag only bypasses pending benchmark clearance here:
    # this isolated input view contains reviewed episodes and zero raw records.
    export(view,destination,allow_unchecked=True)
    info=json.loads((destination/'dataset_info.json').read_text())
    component_counts={}
    for name,entry in info.items():
        examples=json.loads((destination/entry['file_name']).read_text())
        combined.setdefault(name,[]).extend(examples);descriptor[name]=entry
        component_counts[name]=len(examples)
    components[index]['export_counts']=component_counts
if combined.get('efa_atoms'):raise ValueError('Unreviewed raw QA unexpectedly reached this export')
episodes=[row for rows in selected for row in rows]
steps=sum(len(row['steps']) for row in episodes)
visual=sum(atom['kind']=='visual' for row in episodes for step in row['steps'] for atom in step['atoms'])
premises=sum(atom['kind']=='premise' for row in episodes for step in row['steps'] for atom in step['atoms'])
bindings=sum(bool(atom.get('native_evidence',{}).get('object_ids') and row.get('native_program_objects'))
    for row in episodes for step in row['steps'] for atom in step['atoms'] if atom['kind']=='visual')
expected=dict(efa_steps=steps,efa_reads=visual,efa_regions=visual,efa_premises=premises,efa_bindings=bindings)
if any(len(combined.get(name,[]))!=count for name,count in expected.items()):
    raise ValueError('Exported task counts differ from eligible reviewed annotations')
for name,examples in combined.items():
    (dataset/(name+'.json')).write_text(json.dumps(examples),encoding='utf-8')
weighted=combined['efa_steps']*4
(dataset/'efa_steps_weighted.json').write_text(json.dumps(weighted),encoding='utf-8')
descriptor['efa_steps_weighted']=dict(descriptor['efa_steps'],file_name='efa_steps_weighted.json')
(dataset/'dataset_info.json').write_text(json.dumps(descriptor,indent=2),encoding='utf-8')
tasks=['efa_steps_weighted']+[name for name in ['efa_reads','efa_regions','efa_premises','efa_bindings'] if expected[name]]
samples=len(weighted)+sum(expected[name] for name in tasks if name!='efa_steps_weighted')
# Transformers 4.51.3 flushes the partial accumulation batch at each epoch end,
# while its epoch-only max-step estimate uses floor. Explicit ceil preserves
# two complete passes when this one-GPU, batch-size-one dataset has odd length.
updates_per_epoch=math.ceil(samples/2);updates=2*updates_per_epoch
manifest=dict(kind='codex_reviewed_only_sft_export',components=components,train_episodes=len(episodes),
    train_images=len(seen_train_images),episode_ids=sorted(seen_ids),counts=expected,
    step_sampling_multiplier=4,weighted_samples=samples,full_epochs=2,
    per_device_train_batch_size=1,gradient_accumulation_steps=2,
    expected_updates_per_epoch=updates_per_epoch,expected_optimizer_updates=updates,
    source_images_cross_split_checked=True,duplicate_training_ids_and_images_checked=True,
    original_image_pixels_checked=True,training_annotation_file='reviewed_episodes.jsonl',
    native_or_teacher_episode_inputs=False,raw_qa_inputs=False,engineering_only=True,
    benchmark_dedup_status='pending',label_basis=LABEL_BASIS,
    scope='positive_step_route_read_premise_and_existing_binding_sft_not_student_correction_training',
    parent_adapter=str(adapter),parent_adapter_sha256=parent_sha,
    create_new_adapter=False,resume_from_checkpoint=None,optimizer_restarted=True)
(dataset/'export_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
config=yaml.safe_load((root/'code/configs/sft_3b.yaml').read_text())
config.update(model_name_or_path=str(root/'models/Qwen2-VL-2B-Instruct'),
    adapter_name_or_path=str(adapter),create_new_adapter=False,resume_from_checkpoint=None,
    dataset=','.join(tasks),dataset_dir=str(dataset),mix_strategy='concat',packing=False,
    output_dir=str(output),overwrite_output_dir=False,cutoff_len=2048,image_max_pixels=50176,
    lora_rank=16,lora_alpha=32,lora_target='q_proj,k_proj,v_proj,o_proj',
    freeze_vision_tower=True,freeze_multi_modal_projector=True,quantization_bit=4,quantization_method='bnb',
    per_device_train_batch_size=1,gradient_accumulation_steps=2,dataloader_drop_last=False,
    learning_rate=2e-5,num_train_epochs=2.0,max_steps=updates,save_steps=updates_per_epoch,
    seed=2027,data_seed=2027)
config_path.write_text(yaml.safe_dump(config),encoding='utf-8')
print(json.dumps(manifest),flush=True)
PY
python -m efa.sft_entry "$EFA_ROOT/runs/codex_sft_2b_v0.yaml"
python - <<'PY'
import hashlib
import json
from pathlib import Path
root=Path('/data_cinema/zqc/VBC')
manifest=json.loads((root/'data/derived/codex_sft_2b_v0/export_manifest.json').read_text())
original=Path(manifest['parent_adapter'])/'adapter_model.safetensors'
if hashlib.sha256(original.read_bytes()).hexdigest()!=manifest['parent_adapter_sha256']:
    raise RuntimeError('Parent adapter bytes changed during continuation')
state=json.loads((root/'runs/codex_sft_2b_v0/trainer_state.json').read_text())
if state.get('global_step')!=manifest['expected_optimizer_updates'] or abs(state.get('epoch',0)-2.0)>1e-6:
    raise RuntimeError('Training did not complete exactly two full exported-data epochs')
print(json.dumps(dict(completed=True,global_step=state['global_step'],epoch=state['epoch'],
    parent_adapter_unchanged=True,new_adapter='runs/codex_sft_2b_v0')),flush=True)
PY
