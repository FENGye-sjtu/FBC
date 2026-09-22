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
34 focused local tests pass.
Code snapshots and per-case records preserve experiment provenance.
The report parser now also recognizes a single colon-labeled MCQ answer (e.g.
`A: 2 green lights`) and handles empty predictions. Apply the same parser to all
methods after completion; never rewrite raw outputs or use references to extract.

Parser v3 additionally handles an explicit concluding option followed by
explanation and uniquely maps a concluding option phrase back to its declared
label. On fixed outputs it changes reasoned CoT from 522/765 to 524/765 (68.50%)
and the first 178 FaithAct cases from 117 to 122, without changing generations.

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
not repeated unverified values, to downstream prompts.

Completed v4: online 4/4 and CF 4/4, each with one uncertain answer. V5 targeted
case-2 regression: both correct B with explicit uncertainty, online 126.37 s,
CF 308.32 s (v4: 414.17 / 425.01 s). These are shared-hardware debugging times,
not a speed benchmark. V5's exact-claim repeat guard did NOT fire: reworded
claims still reopened the same question. V6 remembered every normalized evidence
question but prematurely finalized its real case-2 regression; both variants
regressed to wrong A. V7 keeps the no-reopen invariant and redirects twice toward
independent evidence or a conditional deduction. Its case-2 trace exercised that
path, but the model ignored both redirects and both variants still answered A.
The 16-case v7 online result is 9/16 versus v5's 11/16 (no wins, losses on cases
2 and 6). V7 CF is also 9/16 with identical answers, but raises uncertainty from
7 to 11 cases and mean latency from 84.66 s to 125.42 s. A v8 attempt to hide
exhausted-question text during redirection was 2/5, identical to v7 on those
cases but much slower, and is rejected. This is deterministic question matching,
not a semantic paraphrase guarantee.

## Completed BF16 packaging diagnostic

Fresh inference with the official VLMEvalKit RealWorldQA MCQ packaging completed
all 765 cases: 539/765 = 70.46% under both the reference-blind final-answer parser
and official AGS rule scorer, versus the paper's 70.1%. This is not yet CoT parity:
the embedded "Please select" directive suppressed reasoning in most outputs.
Current VLMEvalKit exact
matching reports 513/765 = 67.06%; its 26 disagreements all reject duplicated
first-line/final explicit option letters as `Z`. Original xai packaging was
479/765 = 62.61% (AGS 478/765). See `REPLICATION.md`; do not mix these protocols.

Two 128-case BF16 prompt candidates are queued behind full FaithAct. The paired
first-128 targets are 91/128 for directive-dominated selection and 87/128 for
reasoned-v2. A candidate advances only if it preserves near-direct accuracy while
producing substantially more inspectable evidence units.

With the aligned VLMEvalKit first-16 packaging, NF4 direct is 9/16 and reasoned
CoT is 10/16. V7 online and online-CF are both 7/16 with identical answers; CF
executes 20 interventions but raises uncertainty without correcting an answer.
V9 prevents an unverified final synthesis from overwriting the initial candidate
when no later fact was verified. V10 adds reference-blind option-counterfactual
scoring. Their runs are separate and no result is claimed before completion.

## Live work (relative to /data_cinema/zqc/CVPR)

- `runtime/queues/reasoned_bf16`: a clean 765-case CoT run removes the conflicting
  direct-answer directive and completed at 522/765 (AGS 523/765). Its active
  FaithAct job reuses that hashed output.
  The earlier 269-case directive-dominated FaithAct diagnostic is preserved and
  stopped. Log `logs/reasoned_bf16_v2.log`.
- `runtime/queues/pilot16_v7`: completed 16-case online/online-CF comparison;
  both are 9/16. Log `logs/pilot16_v7.log`.
- `runs/eval_protocols`: pinned AGS and VLMEvalKit conversion, per-case decisions,
  source hashes, and score outputs for both inference packagings.
- BF16 v1 FAILED before predictions because automatic placement offloaded to
  disk. V2 FAILED because a root device-map entry moved sharded modules onto one
  GPU during dispatch. V3 removes that root entry and is operational.
- Old NF4 full baseline was intentionally stopped at 107 CoT cases to free memory
  for BF16. Results are preserved, but this is not a completed reproduction.
- Old 3B one-case online/CF diagnostic completed; its worker was stopped.
  The unstarted four-case 3B request is deferred, not failed or completed.

## Continue from here

Read the v7/v8 comparisons and clean reasoned CoT/FaithAct run. Compare with
paper Table 2 FaithAct 74.5% only after all 765 cases complete. Then freeze the
stepwise method and run full RealWorldQA plus MMStar.

Completed v3 raw traces, code snapshots and corrected report were published to
HF `trainingfree/milestones/20260922_pilot4_v3`, commit
`e0db220bee7f1527965cbf2963e77363dbf4162c`. GitHub code publication is batched at
this milestone. Preserve other-computer changes. No current score justifies a superiority
claim. `paper/positioning.md` notes that FaithAct's paper already discusses
online verification, unlike the structure of its released Qwen script.
