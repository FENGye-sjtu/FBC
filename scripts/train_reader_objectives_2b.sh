#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
test -f "$EFA_ROOT/runs/gqa_heldout_2b_v2/summary.json"
test -f "$EFA_ROOT/data/derived/gqa_reader_objectives_v1/reader_objectives.jsonl"
# GPU 0 had a transient competing allocation during v0. Use the observed
# steadier GPU 1, and recheck its available memory before this retry.
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$1==1 && $2>=5000 {print $1}')
if [ -z "$gpu" ]; then echo 'WAIT_FOR_GPU_1_5000_MIB_FREE'; exit 3; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m efa.train_reader_objectives --model "$EFA_ROOT/models/Qwen2-VL-2B-Instruct" --adapter "$EFA_ROOT/runs/gqa_sft_2b_v2" --resampler "$EFA_ROOT/runs/latent_gqa_2b_v2/resampler.pt" --root "$EFA_ROOT/data/derived/gqa_reader_objectives_v1" --output "$EFA_ROOT/runs/reader_objectives_2b_v1" --steps 768 --engineering-only
python -m efa.evaluate_episodes --model "$EFA_ROOT/models/Qwen2-VL-2B-Instruct" --adapter "$EFA_ROOT/runs/gqa_sft_2b_v2" --resampler "$EFA_ROOT/runs/reader_objectives_2b_v1/resampler.pt" --root "$EFA_ROOT/data/derived/gqa_native_review_v2_complete" --output "$EFA_ROOT/runs/gqa_reader_objectives_eval_v1" --policy "$EFA_ROOT/code/configs/policy_engineering.json" --split dev --limit 3 --engineering-only
