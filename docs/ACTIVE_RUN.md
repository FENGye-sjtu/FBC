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

## Completed mixed SFT and matching reader

`runs/mixed_program_sft_2b_v0` completed all 2400 updates in 2141.942 seconds,
with training loss 0.2016731807485246 and epoch 0.744301442084044. Its
precision_policy.json verifies BF16 vision and NF4 language module types.
Both trainer and launcher have terminated; do not restart this completed run.
Authoritative artifacts: all_results.json and trainer_log.jsonl.

`runs/latent_mixed_programs_2b_v0` completed 1660 updates over 830 visual
atoms from 344 episodes, shuffled across sources for two full passes. Runtime
495.9614328859898 seconds; peak allocated bytes 3021314560. The checkpoint and
metrics exist and PID 1076397 has terminated. Source hashes, split checks and
annotation tiers are recorded in input_sources.json. This is positive-only
alignment on the old native/teacher datasets, not the new Codex-reviewed batch.
Do not reuse a reader aligned to another adapter.

Prepared evaluation: scripts/evaluate_mixed_programs_2b.sh uses the same three
GQA dev IDs (9236, 28788, 94674) and three CLEVR dev IDs (10563, 263129, 231696).
Outputs: runs/mixed_gqa_heldout_2b_v0 and runs/mixed_clevr_heldout_2b_v0.
Check scripts/status_mixed_evaluation.sh for actual progress before any launch.
The original uncalibrated thresholds are unchanged; no results are claimed yet.

The fresh 2B run uses DeepEyesV2's LLaMA-Factory SFT, verified mixed precision,
rank-16 LoRA, 2,400 updates and fourfold step sampling. Three sources contribute
344 training trajectories / 3,320 unique examples: 1,043 step examples, 830
reads, 830 routes, 213 premise reads and 404 native-object binding targets.
Sampling yields 6,449 entries. The new binding examples match the runtime
prompt and ordered crops. CLEVR whole-scene counts do not acquire fabricated
single-object binding targets.

## Individually Codex-adjudicated batch

`data/codex_adjudicated_v2` is the current training version, published to
QC1/FBC/efa/codex_adjudicated_v2 at HF commit
`a3ecb54213ee41794ba413c32c164c09ff7bca74`. All 17 candidates were visually
reviewed: 16 accepted, one backpack-count case withheld because a partial bag's
category remains unresolved. Six questions are explicitly revised derivatives.
The 16 accepted episodes have 17 steps (15 one-step, one two-step), with
15 train and one calibration. The train export has 16 reads, 16 routes,
16 step examples and six binding examples: 54 total. Each review retains the
original source, image digest, actual inspection evidence and correction reasons.
These are Codex model annotations, not human gold or measured runtime PASS.

Version v2 supersedes v1 for training: whole-image count routes reuse the already
inspected target as nonempty context, and the plate example has a newly inspected
context including its cup anchor. V1 remains an immutable provenance snapshot.
Both versions remain engineering-only; benchmark decontamination is pending.
The count cases use the complete image for exhaustiveness and do not supply
instance-grounding evidence or deletion negatives. New negative/source-error
variants still need explicit visual review.

The next 48 metadata-selected DeepEyesV2 candidates are listed in
tmp/deepeyes_next48_candidates.json: four OCR/logo, one chart candidate,
12 comparisons, 15 relation/reference, 12 attributes, four multi-entity tasks.
They are not yet image-reviewed or accepted. Upstream source train/dev provenance
is absent, and the comparison references are heavily yes-biased. The chart row
422 requires original-image-only reannotation, not reuse of its external tools.

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
