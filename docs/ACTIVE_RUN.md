# Current experiment handoff

All server paths below are relative to /data_cinema/zqc/VBC. Preserve unrelated
projects and GPU jobs. Read actual artifacts before restarting anything. Recent
SSH attempts timed out before authentication; do not infer a running process from
the prepared launchers. The manuscript/CVPR objective remains incomplete.

## Complete-trajectory continuation prepared

The SFT exporter now retains one complete alternating user/assistant record per
reviewed episode. Its assistant turns contain the visible numbered-step text and
structured atoms. Every episode bundle links its steps to neutral routing targets,
independent source crops, premise reads and available binding targets. This changes
the primary SFT unit without weakening the manuscript's visibility restriction:
the main trace preserves prior dialogue, while routing and evidence reading still
run from fresh neutral-query contexts. No runtime PASS/REVIEW record is fabricated.

Across the three frozen Codex-adjudicated sources, the prepared training set has
75 train episodes/images, 127 natural assistant steps, 106 visual atoms and 30
premise atoms. `scripts/train_full_trajectory_2b.sh` continues the completed
`codex_sft_2b_v0` adapter for three full passes (114 expected optimizer updates),
with all assistant turns supervised (`mask_history=false`) and prompts masked.
`scripts/train_full_evidence_2b.sh` then aligns the compact reader for 424 updates,
trains 16 reviewed negative pairs with all 106 positives replayed for 244 updates,
and evaluates the fixed GQA/CLEVR development cases using preserved multi-turn
history. Neither launcher has run yet; check with `scripts/status_full_trajectory.sh`.

The newly frozen `data/codex_local32_adjudicated_v1` contains 32 reviewed images:
30 accepted episodes and two `needs_review` exclusions, 47 natural steps, and
25 train / two dev / three calibration episodes. Its step distribution is 15
one-step, 13 two-step and two three-step episodes. It is published at HF commit
`5d2f631720daa822390d5679c41ecdc238b03934`; all 234 remote files match local
bytes. These are Codex visual adjudications, not human gold or runtime audits.

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

Completed evaluation: scripts/evaluate_mixed_programs_2b.sh used the same three
GQA dev IDs (9236, 28788, 94674) and three CLEVR dev IDs (10563, 263129, 231696).
Outputs: runs/mixed_gqa_heldout_2b_v0 and runs/mixed_clevr_heldout_2b_v0.
Both summaries and all three result rows per source exist; no evaluation process
remains. Do not restart these runs. The original uncalibrated thresholds are
unchanged. GQA: native reads 8/10, compact 6/10, direct 2/3 strict string,
step-only 2/3, EFA 1/3, zero verified. Direct car answer is 'in car', a semantic
match rejected by the retained strict metric; do not hide this limitation.
EFA answers child/cat/horse; elephant was again changed to horse. CLEVR: native
reads 6/6, compact 2/6, reference-conditioned premise reads 3/3, and direct,
step-only, EFA all 3/3; two EFA finals verified and both correct. All generated
trajectories in both sources still have three steps. There is no audit efficacy
gain demonstrated by these six development cases. Poor compact visual reads
remain a separate problem even where the final yes/no comparison is correct.

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

The next 48 selected DeepEyesV2 original images were read from the existing
10,272,796,390-byte server archive. Three existing decoded-image hashes were
excluded; all remaining 45 candidates were then individually inspected with
their actual target/context crops. Qwen drafted 38 valid episodes and had seven
conversion failures. Published `data/codex_curated45_adjudicated_v1` contains
42 accepted episodes, 79 steps, 35 train / six calibration / one dev, and three
withheld ambiguous cases. Step distribution: 22 one-step, four two-step,
15 three-step, one four-step. Twenty-nine questions are explicitly revised
derivatives. Literal answer-string changes include canonicalization and must
not be reported as semantic corrections. HF commit:
`1beaf3543da3d5a340b50d883f10411800fc5548`. All 272 published artifact hashes
match local bytes; the earlier v2 batch's 67 files were checked at the same head.

Combined Codex-adjudicated data: 62 reviewed candidates, 58 accepted episodes,
96 steps; 50 train / seven calibration / one dev. There are 247 unique train
SFT examples: 86 steps, 69 reads, 69 routes, 17 premise reads, six native-backed
bindings. New DeepEyes examples do not acquire fabricated native object IDs;
explicit Codex binding supervision and corrective source/value variants remain
to be added. Dataset size is still small and formal decontamination is pending.

## Codex-data SFT continuation

Prepared launcher: scripts/train_codex_adjudicated_2b.sh, status via
scripts/status_codex_sft.sh. It continues the completed mixed adapter with a
fresh optimizer into runs/codex_sft_2b_v0; parent weights are never overwritten.
Inputs are only the 50 accepted train trajectories from the two Codex versions,
isolated from raw QA and all native/teacher episode files. Fourfold step sampling
yields 505 entries; 506 optimizer updates cover two complete epochs. Learning
rate 2e-5, context limit 2048, BF16 vision/NF4 language, rank-16 LoRA. This is
positive step/route/read/premise alignment, not completed correction training.
The last successful remote observation showed completion: 506/506 optimizer
updates, epoch 2, runtime 444.2282 seconds; both trainer PID 1090557 and launcher
PID 1090420 had terminated. Precision was BF16 vision / NF4 language. Preserve
this completed run and its parent adapter; do not restart it. The later SSH
rechecks on 2026-09-19 timed out before authentication, so these figures describe
the last observed completed artifact, not a newly polled live process.

Prepared scripts/align_evaluate_codex_2b.sh realigns the reader for 276 updates
over the 69 reviewed visual atoms / 50 training images, then evaluates the fixed
three GQA and three CLEVR development cases with unchanged policy. Its status
script is scripts/status_codex_evaluation.sh. As of the latest recheck, there
is NO confirmed deployment/start/PID for this pipeline; SSH connection timeouts
prevented deployment. Before launching, inspect remote processes/artifacts again.
Do not reuse the old reader for inference with the new Codex adapter.

## Published reviewed corrective supervision

Six accepted DeepEyes training images now have individually reviewed synthetic
error variants: 12 correct-source/wrong-attribute pairs and four wrong-object,
same-attribute pairs. Two missing-anchor crops remain excluded from CE/UL.
The four binding pairs supervise object category only (shirt/skirt and
gloves/ski boots); the shared true colors are never explicit negative targets.
They do not establish full ownership identity or actual student repair success.
Version data/codex_correction_pairs_v1 is published to QC1/FBC at commit
289825b77774bb6940a9b5dc2a0db779578ded34. All 62 remote file hashes match the
local frozen version, including the saved runtime-format and packager sources.
Six original image groups supply 16 pairs; these are not 16 new images.

efa.train_codex_objectives and scripts/train_codex_corrections_2b.sh prepare
joint shared-LoRA/reader training with all 69 reviewed positive reads replayed.
With 16 corrective pairs, two shuffled epochs give 170 optimizer updates.
The adapter learning rate is 1e-6 and reader rate 1e-4; negatives use whole-string
unlikelihood, not exponentiated length-normalized log probabilities. The launcher
requires the completed matching Codex reader and writes new run directories.
No corrective GPU training has been launched or measured yet.

Read-only review of old mixed traces identified a concrete source-repair gap:
GQA 94674's two proposed boxes cover bushes, and binding rejects them. The
controller only expands context, so the target remains wrong; the unverified
fallback ranks horse above the original elephant. This is not a horse PASS.
CLEVR 10563 also changes one count incorrectly while the comparison's yes answer
happens to remain unchanged. Source relocation after failed binding needs an
explicit experiment after the unchanged-policy Codex-data baseline.
The opt-in --reacquire-on-binding-failure implementation is now available in
efa.evaluate_episodes. It only exposes neutral query and previous box geometry
to a fresh original-image route request, records the mode in results/traces,
and preserves invalid status on acquisition failure. It defaults off so the
baseline remains comparable. Related controller and corrective-data/loss tests
pass; this source-relocation experiment has not yet run on the GPU.

## Local data expansion under individual review

data/codex_local32_candidates_v1 stages 32 additional local original images:
16 ChartQA, ten ZwZ, six DeepEyes. Original splits are retained: 27 train,
two dev, three calibration. Exact image hashes were checked against earlier
Codex versions and within the new selection. Qwen produced 27 valid drafts
and five conversion failures; its process completed. Every image and final
source crop is being individually reviewed by Codex before acceptance.
Do not treat the candidate count or teacher draft count as accepted episodes.

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
