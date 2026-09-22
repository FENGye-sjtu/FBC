# CVPR2027: FaithAct-based training-free online verification

Latest user instruction (2026-09-21): prioritize actual accuracy and runnable
experiments. Frozen MLLM; **one step -> evidence audit -> repair or commit -> next
step**. No complete speculative chain followed by cleanup in our method.
After an initial failed audit, repair up to THREE rounds. If still unresolved,
commit an explicit uncertain point and continue, carrying uncertainty into
dependent conclusions. This is not automatic failure/abstention. Backtracking is
deferred. This directory is separate from the older `efa/` SFT
project; it neither changes its files nor interrupts its jobs.

## What is implemented

- `faithact.py`: port of official FaithAct object extraction, GroundingDINO,
  trained CLIP poll, 0.7/0.3 fusion, 0.4/0.6 thresholds and official prompts.
- `core.py`: tentative vs verified vs uncertain steps, three-round repair,
  uncertainty-aware continuation, final deduction audit and complete trace.
- `online.py`: frozen VLM support scoring, separate deduction entailment, optional
  independently localized box deletion versus two equal-area low-overlap controls,
  and candidate-independent answer-consistency scoring (not image CF).
- `online_repair.py`: failure-only independent evidence rereading, with a
  neutral acquisition query and explicitly tentative best-supported fallback.
- `run.py`: same input/decoding settings across CoT, FaithAct, online, online-CF;
  declared CoT prompt variants, reference-blind conclusion/option extraction,
  per-case trace, latency/token counts, and resumable runs.
- `prepare.py`, `queue.py`, `experiment.sh`: pinned asset preparation and one-shot
  resource-aware experiment launch under the authorized CVPR roots only.
- `worker.py`: persistent frozen weights with controller reload between queued
  jobs; `queue_bf16.py` fits unquantized weights across spare GPU memory.
- `report.py`: paired answer accuracy, actual CF executions, accepted repairs,
  uncertain continuation and reference-blind parser corrections from saved traces.
- `compare.py`: the same parser and paired wins/losses across methods stored in
  separate run folders, with raw paths and parser/dataset hashes.
- `eval_protocols.py`: deterministic AGS JSONL conversion, pinned VLMEvalKit
  `can_infer` scoring, and exact RealWorldQA prompt-packaging materialization.
- `answers.py` / `evaluation_suite.py`: frozen reference-blind answer extraction,
  complete-ID checks, immutable input snapshots, and separate unmodified AGS,
  raw VLMEvalKit, and canonical-final VLMEvalKit scores.

Latest fixed-output baseline: 541/765 (70.72%) with the unified parser;
unmodified AGS remains 539/765 (70.46%). This is parsing recovery, not a model
improvement. See `STATUS.md` and `reports/20260922_eval_repair_stage.md`.

The CF MVP uses image-mean masking (not learned causal interventions). Scores and
thresholds are engineering defaults, not calibrated confidence or correctness
guarantees. Insufficient evidence is not silently labeled verified. Exhausted
points continue as uncertain; a bounded finalization produces a best-effort answer
without claiming it passed verification. No claimed score improvement
exists until real experiment outputs support it.

## FaithAct provenance and reproduction targets

Official MIT upstream: https://github.com/lijunxian111/Faithful-First-RPA
Pinned commit: `f0681f4c8d05fa415ee29f3906faa2bdcdf790ab`.
Classifier LFS object: SHA256
`fc4b00a6c00be51e30cdec55f9b9e2c47afe5081082edf93cbdcfef92b68ec5a`,
1,715,124,428 bytes. `prepare.py` verifies the actual downloaded checkpoint.

Paper Table 2, Qwen2.5-VL-7B: RealWorldQA CoT **70.1%**, FaithAct **74.5%**;
MMHal ratings **3.40 / 3.48** (ratings, not accuracy). These are published targets,
**not our measurements**. Primary first replication is all 765 RealWorldQA cases.

Known port choices that must be resolved before declaring paper parity:

See `REPLICATION.md` for the complete known/unknown protocol checklist and
`paper/positioning.md` for the distinction between FaithAct's paper and release.

1. Upstream imports an absent `qwen_inference.py`; our adapter supplies the missing
   model generation implementation. Original raw-CoT prompt/decoding not fully
   released. Our prompt and deterministic decoding are recorded explicitly.
2. Keep Qwen script thresholds 0.25/0.20; paper/Tools.py use 0.35/0.25. Do not hide
   this discrepancy or alter the reproduced baseline based on test accuracy.
3. Share one frozen Qwen instance. The completed reasoned-v2 reconstruction kept
   tools on CPU; the new layout-v5 reconstruction records a separate float32
   tool GPU, with BF16 weights sharded over other free memory. The official full
   classifier checkpoint strictly loads both CLIP encoders
   and the head; no duplicate encoder-weight download is necessary. Numerical
   device differences and token/pixel budgets are logged.
4. Preserve sigmoid(logit_yes), blank-line first/last-step exclusion, and empty
   negative-object descriptions as upstream actually implements them.
5. Report our final-answer parser, AGS rule extraction and VLMEvalKit exact
   matching side by side. The latter has 26 known false negatives on repeated
   first-line/final option letters in the completed 765-case diagnostic. Do not
   compare FaithEvi F-chain values with task accuracy.

## Commands

Local CPU controller tests: `python -m unittest trainingfree.test_core -v`

Server (all new writes stay in `/data_cinema/zqc/CVPR`):

```bash
source /data_cinema/zqc/CVPR/code/trainingfree/server_env.sh
cd "$CVPR_ROOT/code"
# Read-only reuse of installed Python/packages; no installs into the old project.
PY=/data_cinema/zqc/VBC/envs/efa/bin/python
$PY -B -m trainingfree.prepare
$PY -B -m trainingfree.queue --name shared_7b_nf4_uncertain_v1 \
  --min-free-mib 6200 --assets-ready -- \
  bash "$CVPR_ROOT/code/trainingfree/experiment_shared.sh"
```

The queue ignores utilization and shares busy cards when the selected model fits.
NF4 7B pilot uses a 6200 MiB launch floor and 448x448-equivalent pixel budget;
these are engineering settings, not original-precision paper parity. The allocator
is capped to the launch-time free-memory budget minus 256 MiB. No other task is
evicted. BF16 replication also supports `queue_bf16.py` with `--device auto`,
combining launch-time spare memory across busy GPUs without quantization or CPU
offload. Queues expire after 48h. Sync code/data only at research milestones,
not on every implementation edit (latest user instruction).

Next research iterations: inspect the paired pilot's real failures, improve
step/region prompts and gate utility, expand accuracy comparison, then add
dependency-aware rollback. Keep all accepted claims' exposed premises in traces.
