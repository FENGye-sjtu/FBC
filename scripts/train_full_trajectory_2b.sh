#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
python - <<'PY'
import copy
import hashlib
import json
import math
from pathlib import Path
from PIL import Image
import yaml
from efa.export import export
from scripts.package_adjudication import LABEL_BASIS,validate_episode

root=Path('/data_cinema/zqc/VBC')
parent=root/'runs/codex_sft_2b_v0'
dataset=root/'data/derived/codex_full_trajectory_2b_v0'
output=root/'runs/codex_full_trajectory_2b_v0'
config_path=root/'runs/codex_full_trajectory_2b_v0.yaml'
for path in (dataset,output,config_path):
    if path.exists():raise ValueError('Preserve existing artifact: '+str(path))
state=json.loads((parent/'trainer_state.json').read_text())
if state.get('global_step')!=506 or state.get('max_steps')!=506 or abs(state.get('epoch',0)-2.0)>1e-6:
    raise ValueError('The completed reviewed-step parent adapter is required')
for name in ('adapter_model.safetensors','adapter_config.json','all_results.json','precision_policy.json'):
    if not (parent/name).is_file():raise ValueError('Missing parent artifact: '+name)

sources=[root/'data/derived/codex_adjudicated_v2',
         root/'data/derived/codex_curated45_adjudicated_v1',
         root/'data/derived/codex_local32_adjudicated_v1']
selected=[];components=[];ids=set();selected_images=set();train_images=set();heldout_images=set()
for source in sources:
    manifest_path=source/'manifest.json';manifest=json.loads(manifest_path.read_text())
    if not manifest.get('construction_complete') or manifest.get('label_basis')!=LABEL_BASIS:
        raise ValueError('Only completed Codex visual adjudications are allowed')
    registry=[json.loads(line) for line in (source/'raw_tasks.jsonl').read_text().splitlines() if line.strip()]
    for row in registry:
        (train_images if row['split']=='train' else heldout_images).add(row['image_sha256'])
    episodes_path=source/'reviewed_episodes.jsonl'
    episodes=[json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    rows=[]
    for row in episodes:
        if row.get('annotation_status')!='reviewed_episode' or row.get('label_basis')!=LABEL_BASIS:
            raise ValueError('Unexpected annotation tier')
        if row['split']!='train':continue
        if row['id'] in ids or row['image_sha256'] in selected_images:
            raise ValueError('Duplicate training episode or image')
        ids.add(row['id']);selected_images.add(row['image_sha256']);validate_episode(row)
        original=(source/row['image']).resolve()
        with Image.open(original) as image:
            image=image.convert('RGB');digest=hashlib.sha256(image.tobytes()+str(image.size).encode()).hexdigest()
        if digest!=row['image_sha256']:raise ValueError('Training image pixels changed')
        prepared=copy.deepcopy(row);prepared['image']=str(original);rows.append(prepared)
    if len(rows)!=manifest['accepted_split_counts'].get('train',0):raise ValueError('Training count mismatch')
    selected.append(rows)
    components.append(dict(root=str(source),manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        episodes_sha256=hashlib.sha256(episodes_path.read_bytes()).hexdigest(),train_episodes=len(rows)))
if train_images & heldout_images:raise ValueError('Train/held-out image collision')

dataset.mkdir(parents=True)
trajectories=[];bundles=[];descriptor=None
for index,rows in enumerate(selected):
    view=dataset/'reviewed_inputs'/f'source_{index}';view.mkdir(parents=True)
    (view/'raw_tasks.jsonl').write_text('',encoding='utf-8')
    (view/'reviewed_episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    part=dataset/f'source_{index}';export(view,part,allow_unchecked=True)
    exported=json.loads((part/'efa_trajectories.json').read_text())
    linked=[json.loads(line) for line in (part/'efa_episode_bundles.jsonl').read_text().splitlines() if line.strip()]
    if len(exported)!=len(rows) or len(linked)!=len(rows):raise ValueError('Episode export count mismatch')
    trajectories.extend(exported);bundles.extend(linked)
    descriptor=json.loads((part/'dataset_info.json').read_text())['efa_trajectories']
    components[index]['turns']=sum(len(row['messages'])//2 for row in exported)
steps=sum(len(row['steps']) for rows in selected for row in rows)
visual=sum(atom['kind']=='visual' for rows in selected for row in rows for step in row['steps'] for atom in step['atoms'])
premises=sum(atom['kind']=='premise' for rows in selected for row in rows for step in row['steps'] for atom in step['atoms'])
if (len(trajectories),steps,visual,premises)!=(75,127,106,30):
    raise ValueError('Reviewed source composition changed')
if any(len(row['messages'])<2 or len(row['messages'])%2 or
       [m['role'] for m in row['messages']]!=['user','assistant']*(len(row['messages'])//2) or
       sum(m['content'].count('<image>') for m in row['messages'])!=1 for row in trajectories):
    raise ValueError('Invalid complete multi-turn serialization')
(dataset/'efa_trajectories.json').write_text(json.dumps(trajectories),encoding='utf-8')
(dataset/'efa_episode_bundles.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in bundles),encoding='utf-8')
descriptor=dict(descriptor,file_name='efa_trajectories.json')
(dataset/'dataset_info.json').write_text(json.dumps({'efa_trajectories':descriptor},indent=2),encoding='utf-8')
updates_per_epoch=math.ceil(len(trajectories)/2);updates=updates_per_epoch*3
run_manifest=dict(kind='complete_multiturn_episode_sft',components=components,train_episodes=75,
    train_images=75,steps=steps,visual_atoms=visual,premise_atoms=premises,
    sft_examples=len(trajectories),assistant_turns=steps,full_epochs=3,
    per_device_train_batch_size=1,gradient_accumulation_steps=2,
    expected_updates_per_epoch=updates_per_epoch,expected_optimizer_updates=updates,
    train_on_prompt=False,mask_history=False,packing=False,
    main_trace_visibility='full_prior_user_assistant_history',
    evidence_visibility='separate_neutral_query_forwards_linked_by_episode_bundle',
    runtime_audited_episodes=0,actual_student_repairs=0,engineering_only=True,
    benchmark_dedup_status='pending',label_basis=LABEL_BASIS,
    parent_adapter=str(parent),parent_adapter_sha256=hashlib.sha256((parent/'adapter_model.safetensors').read_bytes()).hexdigest())
(dataset/'export_manifest.json').write_text(json.dumps(run_manifest,indent=2),encoding='utf-8')
config=yaml.safe_load((root/'code/configs/sft_3b.yaml').read_text())
config.update(model_name_or_path=str(root/'models/Qwen2-VL-2B-Instruct'),adapter_name_or_path=str(parent),
    create_new_adapter=False,resume_from_checkpoint=None,dataset='efa_trajectories',dataset_dir=str(dataset),
    mix_strategy='concat',packing=False,train_on_prompt=False,mask_history=False,
    output_dir=str(output),overwrite_output_dir=False,cutoff_len=4096,image_max_pixels=50176,
    lora_rank=16,lora_alpha=32,lora_target='q_proj,k_proj,v_proj,o_proj',
    freeze_vision_tower=True,freeze_multi_modal_projector=True,quantization_bit=4,quantization_method='bnb',
    per_device_train_batch_size=1,gradient_accumulation_steps=2,dataloader_drop_last=False,
    learning_rate=1e-5,num_train_epochs=3.0,max_steps=updates,save_steps=updates_per_epoch,
    seed=2027,data_seed=2027)
config_path.write_text(yaml.safe_dump(config),encoding='utf-8')
print(json.dumps(run_manifest),flush=True)
PY
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=7000 {print $1,$2}' | sort -k2nr | head -n1 | awk '{print $1}')
if [ -z "$gpu" ]; then echo 'WAIT_FOR_7000_MIB_FREE'; exit 3; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -i "$gpu" --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader
python -m efa.sft_entry "$EFA_ROOT/runs/codex_full_trajectory_2b_v0.yaml"
python - <<'PY'
import json
from pathlib import Path
root=Path('/data_cinema/zqc/VBC');run=root/'runs/codex_full_trajectory_2b_v0'
manifest=json.loads((root/'data/derived/codex_full_trajectory_2b_v0/export_manifest.json').read_text())
state=json.loads((run/'trainer_state.json').read_text())
if state.get('global_step')!=manifest['expected_optimizer_updates'] or not (run/'adapter_model.safetensors').is_file():
    raise RuntimeError('Complete-trajectory SFT did not finish')
print(json.dumps(dict(completed=True,global_step=state['global_step'],epoch=state.get('epoch'))),flush=True)
PY
