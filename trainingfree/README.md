# CVPR2027: FaithAct-based training-free online verification

Latest user instruction (2026-09-21): prioritize actual accuracy and runnable
experiments. Frozen MLLM; **one step -> evidence audit -> repair or commit -> next
step**. No complete speculative chain followed by cleanup in our method.
Backtracking is deferred. This directory is separate from the older `efa/` SFT
project; it neither changes its files nor interrupts its jobs.

## What is implemented

- `faithact.py`: port of official FaithAct object extraction, GroundingDINO,
  trained CLIP poll, 0.7/0.3 fusion, 0.4/0.6 thresholds and official prompts.
- `core.py`: tentative vs committed steps, failure-only repair, bounded budgets,
  accepted-premise-only continuation, final deduction audit and complete trace.
- `online.py`: frozen VLM support scoring, separate deduction entailment, optional
  independently localized box deletion versus two equal-area low-overlap controls.
- `run.py`: same input/decoding settings across CoT, FaithAct, online, online-CF;
  per-case trace, latency/token counts, resumable runs, exact-match accuracy.
- `prepare.py`, `queue.py`, `experiment.sh`: pinned asset preparation and one-shot
  resource-aware experiment launch under the authorized CVPR roots only.

The CF MVP uses image-mean masking (not learned causal interventions). Scores and
thresholds are engineering defaults, not calibrated confidence or correctness
guarantees. Insufficient evidence is not silently accepted. Exhausted cases are
unanswered and counted wrong in end-task accuracy. No claimed score improvement
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

1. Upstream imports an absent `qwen_inference.py`; our adapter supplies the missing
   model generation implementation. Original raw-CoT prompt/decoding not fully
   released. Our prompt and deterministic decoding are recorded explicitly.
2. Keep Qwen script thresholds 0.25/0.20; paper/Tools.py use 0.35/0.25. Do not hide
   this discrepancy or alter the reproduced baseline based on test accuracy.
3. Share one frozen Qwen instance and keep tools resident on CPU to fit a 24 GB
   GPU. The official full classifier checkpoint strictly loads both CLIP encoders
   and the head; no duplicate encoder-weight download is necessary. Numerical
   device differences and token/pixel budgets are logged.
4. Preserve sigmoid(logit_yes), blank-line first/last-step exclusion, and empty
   negative-object descriptions as upstream actually implements them.
5. Our strict normalized exact-match is an initial metric. Validate against the
   official dataset evaluation protocol before calling scores directly matched.
   Do not compare FaithEvi F-chain values with task accuracy.

## Commands

Local CPU controller tests: `python -m unittest trainingfree.test_core -v`

Server (all new writes stay in `/data_cinema/zqc/CVPR`):

```bash
source /data_cinema/zqc/CVPR/code/trainingfree/server_env.sh
cd "$CVPR_ROOT/code"
# Read-only reuse of installed Python/packages; no installs into the old project.
PY=/data_cinema/zqc/VBC/envs/efa/bin/python
$PY -B -m trainingfree.prepare
$PY -B -m trainingfree.queue --name faithact_reproduction_v1 --assets-ready -- \
  bash "$CVPR_ROOT/code/trainingfree/experiment.sh"
```

The queue does not reserve or evict anyone else's GPU. It waits for >=22 GB free
and <=5% utilization, checks again under a project lock, and exits after 48h if
unavailable. Pilot failures stop progression, not all future research.

Next research iterations: inspect the paired pilot's real failures, improve
step/region prompts and gate utility, expand accuracy comparison, then add
dependency-aware rollback. Keep all accepted claims' exposed premises in traces.
