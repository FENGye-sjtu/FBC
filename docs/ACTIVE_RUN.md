# Current experiment handoff

All server paths below are relative to /data_cinema/zqc/VBC. SSH is working.
Preserve unrelated projects and GPU jobs. Read actual artifacts before restarting
anything. The manuscript/CVPR objective remains incomplete.

## Completed reader comparison

- Baseline: gqa_sft_2b_v2 (1,200 updates), latent_gqa_2b_v2 (768 updates),
  gqa_heldout_2b_v2 (three dev cases). Verified BF16 vision / NF4 language.
- New run: reader_objectives_2b_v1 completed 768 updates in 489.317 seconds;
  peak allocation 3,085,314,048 bytes. Frozen shared adapter; summed read NLL,
  0.2 identity NLL and 1.0 negative unlikelihood. Mean bundled loss 3.244506.
- Same three dev cases, same uncalibrated policy: native reads 8/10, compact
  reads 6/10, direct QA 2/3, unaudited steps 1/3, EFA 1/3. These aggregate
  scores are unchanged. All generated trajectories contain three steps.
- Car case: EFA still answers man, now with effective verified=true. Its
  generated queries drift from location to a person's category; verification
  does not establish that the claims answer the original question.
- Cat case: EFA changes from bear to cat, still unverified.
- Elephant case: EFA changes from elephant to horse. The final incumbent was
  elephant; deletion/specificity rejected it, horse passed local repair, and
  the final inherited unverified status from the field-existence step.
- Thus there is no demonstrated audit advantage. One of three outputs is
  labeled verified and that output is wrong. Do not report coverage as accuracy.
- Full traces, losses, metrics and policy are frozen in
  data/derived/gqa_reader_objectives_eval_v1 and local data of the same name.
  All downloaded file hashes match. HF commit:
  c88b458f931007b4ad82b3a4d23de30977b0ef72.
- The earlier GPU-0 v0 OOM logs remain preserved. GPU-1 v1 finished; no other
  job was killed or modified. Do not restart either experiment.

## Next mixed SFT

Server shell PID 1054370 runs tools/continue_mixed_programs_v0.sh. It waits for
the completed reader evaluation, packages it, then runs the mixed SFT launcher.
Log: logs/continue_mixed_programs_v0.log. Check scripts/status_mixed_programs.sh
for actual optimizer progress. Training target: runs/mixed_program_sft_2b_v0.
Do not submit a duplicate. A missing free GPU or failure exits this pipeline;
the existence of the shell launch is not evidence of completed training.
Latest observation: training PID 1054891 on GPU 0 completed update 76/2400.
Its precision_policy.json verifies BF16 vision and NF4 language module types.
The trainer estimated about 36 minutes remaining at that point; this is not a
completion guarantee. No new adapter inference or mixed evaluation has run yet.

The fresh 2B run uses DeepEyesV2's LLaMA-Factory SFT, verified mixed precision,
rank-16 LoRA, 2,400 updates and fourfold step sampling. Three sources contribute
344 training trajectories / 3,320 unique examples: 1,043 step examples, 830
reads, 830 routes, 213 premise reads and 404 native-object binding targets.
Sampling yields 6,449 entries. The new binding examples match the runtime
prompt and ordered crops. CLEVR whole-scene counts do not acquire fabricated
single-object binding targets. Compact-reader alignment for this new adapter
must follow training; the previous adapter's reader cannot be reused at inference.

## Data frozen and uploaded

- GQA reviewed base: 156 approvals / 512 candidates; 126 train / 11 dev /
  19 calibration. HF e19bda2b491706c57cabe7873767c7258eb6311a.
- Native reader objectives: 384 train records, 54 category negatives on 27 atoms.
  HF 6114f79c0c3b3b521c63cf1140b11ad26325e2f9.
- CLEVR: 256 native episodes, 213 train / 27 dev / 16 calibration.
  HF 3d694fb95e7a7f52de40e2e0c3a59a19ac04fa40.
- New four-step candidates: 64 original images, no exact-image overlap with
  prior GQA source; HF 06d89d5f0ae8956a2179d4b02808d605a30eb192.
- Completed four-step review subset: 63 reviewed, six approvals (five train /
  one dev), 57 rejections. One API data_inspection_failed case remains excluded
  after retry. data/derived/gqa_native_long_review_v1_completed63, HF
  ed8b84c172ca85a2f2295495cf264d4a481d61a3. No teacher job remains active.

## Remaining research

Train reliable task-relevant step generation and learned source binding/reading;
construct wrong/incomplete source supervision; collect real student repair and
parent-revisit trajectories. Four-step supervision remains scarce. Calibration,
near-duplicate/benchmark clearance, matched comparisons and paper-scale results
are unfinished. All current runs are explicitly engineering-only.
