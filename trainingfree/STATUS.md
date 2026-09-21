# Training-free CVPR2027 status — 2026-09-22

Implemented research prototype with measured outputs, not a completed paper or
a successful replication claim. Read live files, not process-launch status.

## Fixed scope

Frozen MLLM; one step then audit; initial audit plus at most THREE repair rounds;
unresolved evidence stays explicitly uncertain and reasoning continues.
A verified visual fact survives a failed attached answer: withhold the answer,
commit only the fact, and derive a corrected conclusion next.
All remote writes stay under the two CVPR roots. VBC/covt resources are read-only.
Share spare GPU memory regardless of utilization; never evict others' jobs.
Batch publication at meaningful milestones, not every edit.

## Verified implementation

Official FaithAct port/checkpoint and all 765 original RealWorldQA cases are ready.
Implemented structured atomic steps, processed-image coordinate conversion,
crop reacquisition, support/entailment gates, CF masking with matched controls,
uncertain continuation, persistent workers, and explicit BF16 multi-GPU packing.
24 local and remote tests pass.
Code snapshots and per-case records preserve experiment provenance.

## Measurements (development only)

Run `realworldqa_7b_nf4_pilot4_stepfix_v3`: CoT 3/4 after reference-blind
terminal-option parser correction; FaithAct 2/4; online 4/4. One online answer
used uncertain fallback. CF completed at 3/4, also with one uncertain fallback;
it did NOT outperform online. Its 26 actual counterfactual audits are recorded.
Do not infer benchmark gains from four development examples, or mix versions.
The older FaithAct 4/4 used a different prompt/run and is not pooled here.

Case 2 revealed that a supported yellow-light observation was discarded when its
answer failed entailment. Later drafts repeated a red-light claim. Version v4
preserves the observation and withholds the invalid conclusion. The v3 numbers
predate this fix. Original traces remain unchanged; reports show parser updates.
V4 still repeated an exhausted point despite prompt instructions. V5 therefore
enforces the point budget in the controller and exposes unresolved questions,
not repeated unverified values, to downstream prompts. Do not claim v5 efficacy
until its real-model regression and subsequent 16-case run finish.

## Live work (relative to /data_cinema/zqc/CVPR)

- `runtime/workers/7b_dev`: job 002 is the four-way v3 pilot; job 003 runs
  online/CF v4 on four cases; 003a tests v5 on the repeated-point case; 004 is the
  16-case v5 comparison with a declared standardized FaithAct answer-format
  variant. Log `logs/resident_7b_v2.log`.
- `runtime/workers/7b_bf16_replication_v3`: job 001 targets all 765 CoT and
  FaithAct examples, BF16, 1003520 max pixels, 512 output tokens.
  Log `logs/bf16_replication_v3.log`. Real predictions are now being produced;
  `runtime_model.json` confirms all 729 parameter tensors are torch.bfloat16
  and all placements are GPUs. The full reproduction score is still pending.
  Do not change `faithact.py` or pinned upstream templates during this run: its
  older harness imports the FaithAct class only after the CoT phase completes.
- BF16 v1 FAILED before predictions because automatic placement offloaded to
  disk. V2 FAILED because a root device-map entry moved sharded modules onto one
  GPU during dispatch. V3 removes that root entry and is operational.
- Old NF4 full baseline was intentionally stopped at 107 CoT cases to free memory
  for BF16. Results are preserved, but this is not a completed reproduction.
- Old 3B one-case online/CF diagnostic completed; its worker was stopped.
  The unstarted four-case 3B request is deferred, not failed or completed.

## Continue from here

Complete/read v3 and v4 traces; verify BF16 runtime, then let the complete baseline
finish. Compare with paper Table 2 CoT 70.1% / FaithAct 74.5% only with disclosed
protocol differences in `REPLICATION.md`. Extend paired development to 16 cases,
freeze the method, and run full evaluation plus MMStar.

Completed v3 raw traces, code snapshots and corrected report were published to
HF `trainingfree/milestones/20260922_pilot4_v3`, commit
`e0db220bee7f1527965cbf2963e77363dbf4162c`. GitHub code publication is batched at
this milestone. Preserve other-computer changes. No current score justifies a superiority
claim. `paper/positioning.md` notes that FaithAct's paper already discusses
online verification, unlike the structure of its released Qwen script.
