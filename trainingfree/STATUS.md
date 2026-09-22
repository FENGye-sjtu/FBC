# Training-free CVPR2027 status — 2026-09-22

Implemented research prototype with measured outputs, not a completed paper or
a successful replication claim. Read live files, not process-launch status.

## Current measured results

The server evaluation suite has been rerun with `committed_answer_v3` on all
three 765-case BF16 output files:

| Output | New reference-blind parser | AGS raw output | VLMEvalKit raw output | VLMEvalKit canonical final |
|---|---:|---:|---:|---:|
| directive-dominated | 541/765, 70.72% | 539/765, 70.46% | 513/765, 67.06% | 541/765, 70.72% |
| reasoned-v2 prompt | 526/765, 68.76% | 523/765, 68.37% | 456/765, 59.61% | 526/765, 68.76% |
| FaithAct, paired with reasoned-v2 | 528/765, 69.02% | 461/765, 60.26% | 513/765, 67.06% | 528/765, 69.02% |

The earlier baseline increases to 70.72% and 68.76% were **fixed-output parser
recoveries, not model improvements**. These complete-file scores are unchanged
by `committed_answer_v3`. Canonical final
means that VLMEvalKit receives the reference-blind extracted answer; it is not
independent evidence of a model gain. Raw outputs remain unchanged. Current
full suite reports:
[directive](reports/20260922_parser_v3_directive_eval.json),
[reasoned-v2](reports/20260922_parser_v3_reasoned_v2_eval.json),
[FaithAct](reports/20260922_parser_v3_faithact_reasoned_v2_eval.json).

Parser v3 is frozen at SHA256
`49c5a365e342cf423be14fb4ef9d59e49e1bcf4f98f1036f101f0a71c1140a83`.
Its additive rule handles previously unresolved explicit final-line labels;
it does not change labels that v2 already resolved. The audit covers 20 files
and 3,794 rows, with zero changes to already-valid v2 labels. It retains v2's
semantic limitations and is not a perfect interpretation of arbitrary prose.
The three full-765 and development-prompt scores are unchanged. All tools use
the shared source rule: non-null `raw` has priority, explicit empty `raw` remains
empty, and absent/null `raw` falls back to `answer`. Preserve original outputs
and report parser version, source hash, and each upstream score separately.

FaithAct corrected 33 baseline answers and degraded 31: net +2/765 (+0.26
percentage points; McNemar p=0.901). The 141 cases with tool records gained one
answer overall; the 114 cases whose final prompt actually included box text
had 14 corrections and 14 regressions, net zero. Noun extraction recorded zero
format errors across 212 calls. See the [paired report](reports/20260922_faithact_paired.json)
and [coverage diagnosis](reports/20260922_faithact_diagnostics.json).
This reconstruction is 5.48 percentage points below the paper's 74.5%; it does
not establish exact protocol parity or a reliable tool-induced accuracy gain.

Reasoned-v2 is still mostly answer-only: 602/765 outputs contain at most two
whitespace-delimited words, and only 141/765 have an interior paragraph eligible
for FaithAct checking. The score gap does not establish that genuine CoT hurts
accuracy. This limited input-step coverage also constrains the FaithAct result.

V13 online independent rereading completed at 8/16, below matched NF4 direct
9/16 and CoT 10/16. Among 15 valid initial answers it corrected IDs 1/8 but
degraded 9/12/15; ID 7 had no valid initial answer and finished correct.
It made 21 logged rereads, with 6 uncertain final answers and no runtime errors.
V13 CF completed at 7/16 with 9 uncertain finals and 16 actual CF checks
(48 masked-image calls). It corrected ID 8 but degraded 9/12/15 among valid
initial answers. V14 has also completed: online 8/16 and CF 7/16, matching
V13's scores. The compact initial prompt has not increased final accuracy on
this development subset. These results do not support a CF benefit.
V15 online has now completed at 9/16, versus V14's 8/16 and CoT's 10/16.
Relative to V14 it recovers IDs 9/12 and loses ID 7, net +1. Within V15 itself,
the 14 cases with valid initial answers go from 9 initial correct to 8 final
correct: zero corrections and one regression, ID 7. Initial-answer-missing
IDs 6/9 are reported separately and are not counted as corrections. V15 has
7 uncertain finals, zero runtime errors, and mean latency 95.448 seconds on
shared hardware. See the [paired comparison](reports/20260922_v15_paired.json)
and [online diagnostics](reports/20260922_v15_online_diagnostics.json).
V9 CF was also 8/16. V10's first-four smoke was 1/4;
its separate case-9 test was wrong. Its full-16 job 006 was stopped and partial
records are preserved. These are development results, not benchmark gains.
V7 used `max_pixels=200704`, whereas V9/V10 and the paired controls used
`1003520`; differences between them cannot be attributed to the method alone.

V16 is now complete on the same NF4 16-case development subset: online is
9/16 and CF is 8/16. Online has 8 uncertain/8 model-verified finals; among 14
valid first proposals it has zero gains and one loss (ID 7). CF has 9 uncertain/
7 model-verified finals, zero gains and two losses (IDs 1/7), with 14 target
mask checks and 42 total masked-image calls. The closed-question identity fix
therefore repaired the controller budget but did not improve accuracy on this
subset. See [online](reports/20260922_v16_online_diagnostics.json) and
[CF diagnostics](reports/20260922_v16_online_cf_diagnostics.json).

All three new BF16 prompt comparisons have completed on the same first 128
development cases, using the same committed-answer parser:

| Prompt | Correct | Valid option labels | Outputs with FaithAct-interior evidence |
|---|---:|---:|---:|
| brief-v4 | 89/128, 69.53% | 128/128 | 34/128 |
| adaptive-v4 | 80/128, 62.50% | 121/128 | 69/128 |
| layout-v5 | 90/128, 70.31% | 128/128 | 128/128 |

The direct-selection reference is 92/128 and old reasoned-v2 is 87/128.
Layout-v5 is selected as the candidate combining answer accuracy with complete
inspectable-step coverage among these three prompts. Its 70.31% is a 128-case
development score, not a completed 765-case benchmark score. No literal
placeholders were found in its 128 outputs, and inspected samples contain actual
visual statements; format coverage does not establish factual correctness.
The [layout-v5 evaluation](reports/20260922_parser_v3_prompt_layout128_eval.json) preserves
the measurement and source hashes.

The separate layout-v5 validation slice, IDs 128–255, has completed at
96/128 (75.00%), with 127 outputs containing FaithAct-interior evidence. AGS raw
also scores 96/128; VLMEvalKit raw scores 92/128 (71.875%), and canonical-final
scoring gives 96/128. On these exact IDs, the old direct-selection output is
91/128 and reasoned-v2 is 89/128. Layout-v5 has 11 wins/6 losses versus direct
(net +5, McNemar p=0.3323), and 11 wins/4 losses versus reasoned-v2 (net +7,
p=0.1185). These are promising paired validation results, not statistically
significant or full-765 gains. See the [validation evaluation](reports/20260922_parser_v3_layout_v5_validation128_eval.json)
and [paired results](reports/20260922_validation_paired.json).

FaithAct on the completed layout-v5 development cache has also finished:

| Output / scorer | Correct | Accuracy |
|---|---:|---:|
| GPU-tool FaithAct, committed-answer v3 | 88/128 | 68.75% |
| Same raw output, official AGS | 23/128 | 17.97% |
| Same raw output, VLMEvalKit raw | 68/128 | 53.125% |
| Same output, VLMEvalKit canonical final | 88/128 | 68.75% |

Against its exact cached CoT baseline of 90/128, FaithAct has 10 corrections
and 12 regressions (net -2; McNemar p=0.8318). All 128 cases have tool records,
and 85 include bounding-box evidence. Improved tool coverage has not produced
an accuracy gain in this comparison. Some outputs contain contradictions
between prose and the committed option label; answer-label correctness must
not be interpreted as faithful reasoning. The low official raw-output scores
are retained rather than replaced by the canonical result. See the [complete
scorer suite](reports/20260922_parser_v3_layout_v5_faithact128_eval.json) and
[FaithAct diagnostics](reports/20260922_layout_v5_faithact128_diagnostics.json).

The full 765-case layout-v5 sequence is now complete. Its CoT is 542/765
(70.85%); released-template FaithAct is 510/765 (66.67%) under parser v4,
with 57 corrections, 89 regressions and net -32 (McNemar p=0.0101). Parser v3
gave 509 because of one Markdown-bold marker miss; this is a fixed-output
extraction correction, not a model gain. The 546 cases receiving box text
change 379 to 349 (40 corrections/70 regressions, p=0.00545); 218 tool cases
without box text change 162 to 159. Raw AGS scores 125/765 (16.34%), raw
VLMEvalKit 413/765 (53.99%), and VLMEvalKit canonical-final 510/765 (66.67%).
This is a negative result for our layout-v5 input plus the released rewrite,
not the official-GCOT/paper-appendix runs that follow. See the historical
[v3 scorer suite](reports/20260922_parser_v3_layout_v5_faithact_full765_eval.json)
and [v4 paired diagnostics](reports/20260922_parser_v4_layout_v5_faithact_full765_diagnostics.json).

The released `TEMPLATE_GCOT` candidate has completed all 128 pairs at 78/128
before and 78/128 after FaithAct: 17 corrections and 17 regressions (p=1.0).
All initial chains contain inspectable interiors. Initial AGS/raw-VL/canonical
scores are 77/49/78; rewritten scores are 16/58/78. The same slice's direct
baseline is 92/128, so this released Grounded-CoT prompt should not be mislabeled
as the missing FaithAct Table-2 CoT protocol. See the
[protocol audit](reports/20260922_faithact_prompt_protocol_audit.md).

## Active work

- `runtime/queues/reasoned_bf16`: the released-template FaithAct run is complete.
  Its full raw outputs, evaluator suite and paired diagnostics were published
  to HF `trainingfree/milestones/20260922_eval_repair`, commit
  `15ca6ded503bcecf3b699eb3b73bc58d52c8296d` (261 files). Later-finishing results
  stay on the server until the next batch; see the publication receipt. The
  published cutoff includes the earlier V13-online/brief-v4-128 stage, not every
  later result reported here. Its parser cutoff is the earlier v1 stage; it
  excludes v2/v3 replay and subsequent outputs. Source code and reports are
  committed at research milestones; see
  Git history for publication state.
- `runtime/queues/layout_v5_bf16` has completed job 001 (cached development
  FaithAct), job 002 (IDs 128–255 CoT validation), and job 003 (full 765-case
  layout-v5 CoT/FaithAct). Job 003a (released `TEMPLATE_GCOT` + released
  rewrite/0.25/0.20) and job 003b (same cached GCOT with Appendix-D-style
  rewrite/0.35/0.25) are complete. Job 003c's paper Sec. 3.1 literal
  `Think step by step.` initial CoT is complete at 94/128 (73.44%), versus
  same-ID direct 92, layout-v5 90 and released GCoT 78. All 128 have inspectable
  interiors; raw AGS/raw VL/canonical scores are 35/58/94. Its released-code
  released-code FaithAct phase is complete at 87/128 (67.97%): 12 corrections,
  19 regressions, net -7, p=0.2810. Raw AGS/raw VL/canonical scores are 24/69/87;
  all 128 use tools, with 738 noun inspections and 112 cases receiving box text.
  The released-GCOT/released-rewrite pair is now complete: 78/128 before and
  78/128 after (17 corrections, 17 regressions, p=1.0), versus direct 92/128
  on the same IDs. Initial raw AGS/raw VLMEval/canonical scores are 77/49/78;
  rewritten scores are 16/58/78. The severe AGS raw drop is a known regex-format
  mismatch with `Thus, the final answer is ...`, not the common task-accuracy
  result. Therefore released GCOT is not assumed to be FaithAct's missing
  initial adapter. The paper-appendix rewrite is also 78/128, with one
  correction/one regression versus cached GCoT, valid labels 125/128, and raw
  AGS/raw VL/canonical scores 20/56/78.
  Job 004 then runs V16 online/CF;
  job 005 now runs V19 evidence-first online/CF on the same IDs 128–255. None of
  those BF16 jobs has a score yet.
- The new queue reserves a separate GPU for FaithAct tools. At this launch the
  model weights use physical GPUs 1 and 0 (local CUDA 0 and 1), and tools use
  physical GPU 3 (local CUDA 2). `CVPR_GPU_BUDGETS_MIB` contains only the weight
  devices; `CVPR_TOOL_DEVICE` and `CVPR_TOOL_GPU_BUDGET_MIB` record the independent
  tool placement and startup memory cap. Selection uses free memory and actual
  runtime records, with no utilization gate. Tool networks run in float32 eval
  mode. These GPU-tool runs have separate directories and provenance from the
  completed reasoned-v2 FaithAct run whose tools ran on CPU; they are distinct
  experimental outputs.
- `runtime/queues/v13_cf` has completed V15, V16 and both V17 arms. V17 online
  and CF are each 8/16 versus exact matched NF4 direct 9/16; both have zero
  gains and one loss (ID 12), where its 64-token safety
  anchor became invalid (`None`) while the matched direct output was correct B.
  The queued BF16 protocol now uses the exact direct generation budget, records
  anchor text/answer, advances after an answer-withheld deduction instead of
  repeating it, and skips non-identifying pixel deletion for negative claims.
  That old V17 fix has no BF16 score. Running V18
  `evidence_first_v5_ground_then_commit` additionally borrows FaithAct's useful
  Grounding-to-Observation ordering for each atomic fact and permits a deduction
  to defer (`final_answer=null`) when the verified ledger is not yet sufficient,
  instead of forcing a premature option. V17 stores a direct answer
  only as an uncertain safe anchor, hides answer options from visual proposals
  and repairs, and permits an answer only in a separate ledger-grounded deduction.
  A failed deduction routes to a new option-blind evidence point. See the
  [completed old-protocol online diagnostic](reports/20260922_v17_evidence_first_online_diagnostics.json).
  V18 online is complete at 8/16 versus its exact direct anchor 9/16: zero gains
  and one loss, ID 3. It has 12 uncertain/4 model-verified finals, 73 audits,
  no runtime errors and mean latency 108.80 seconds. Its ID 3 trace exposed a correlated verifier
  error that overwrote a correct direct anchor after only one visual fact. V19 is
  queued next with two fixes: strip answer-bearing suffixes from neutral evidence
  questions, and require two distinct verified visual facts before a deduction
  may change a valid direct anchor. The duplicate old-queue V18 request was
  marked superseded before it could write a second row. V19 replaces V18 in the
  pending BF16 validation job on the same IDs. V18-CF was intentionally skipped
  after the online diagnosis; V19 is now running on the independent spare-memory
  worker instead of spending another arm on a superseded gate.
- V12 was stopped after an empty visual-question field bypassed the exhausted-
  point key; its events are retained. V13 fills that missing key from the neutral
  public question and uses independent rereading during repair. It retains the
  highest-support tentative candidate under explicit uncertainty if needed.
  Scores across different claims are not calibrated probabilities. Both V13
  and V14 arms are now complete, as reported above.
- Existing scorer suites use the `reports/20260922_parser_v3_*_eval.json` prefix,
  with raw snapshots, per-case disagreements, and source hashes. They include
  the completed GPU-tool FaithAct and separate CoT validation slice. The V15
  and validation paired reports also record matched model/input budgets.
  Parser v4 adds only a formatting recovery for a Markdown-bold terminal marker
  such as `**Final Answer**: B`; fixed-output rescoring remains separate from
  model/method gains.

V10's option-likelihood check is answer consistency, not an image counterfactual.
Localized image masking with control regions is the image-intervention component.
See the [stage research record](reports/20260922_eval_repair_stage.md) for the
completed FaithAct analysis and remaining method limitations.

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
The last complete server integration sweep passed 125 tests. After the new
FaithAct protocol and evidence-first changes, 71 directly affected controller,
prompt, analyzer and core tests pass locally; the full sweep awaits the next
milestone environment run because this Windows interpreter lacks PyTorch.
Code snapshots and per-case records preserve experiment provenance.
The current `committed_answer_v4` parser accepts clear final labels, structured
formats, affirmative embedded option labels, and uniquely matched public option
wording while rejecting negated or ambiguous mentions. Reports use the shared
parser source SHA256 `6d7c3d98c5ef949f73a9d982e2503f6cdcd79e89d81062c028247221856fcb8d` and the shared
raw-source rule and record `parser_version`. Online diagnostics with public
cases always re-score the original source; saved correctness is a separate
diagnostic. Initial answers that are missing or invalid are excluded from
correction gains. Apply the parser equally to all methods; never use reference
labels as extraction input or rewrite the original generations. V4 differs from
v3 only by recognizing a Markdown-bold terminal marker before the colon; the
historical v3 reports remain immutable.

## Historical engineering results (development only)

These entries retain earlier configurations and must not replace the current
results or be pooled across packaging, precision and input-budget changes.

An earlier permissive-parser replay, before the committed-answer versions,
changed reasoned CoT from the stored 522/765 to
524/765 (68.50%), and a 178-case FaithAct snapshot from 117 to 122. Those were
formatting recoveries on fixed outputs, superseded by the current suite above.

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

## Historical BF16 packaging diagnostic

Fresh inference with the official VLMEvalKit RealWorldQA MCQ packaging completed
all 765 cases: 539/765 = 70.46% under both the then-current final-answer parser
and official AGS rule scorer, versus the paper's 70.1%. This is not yet CoT parity:
the embedded "Please select" directive suppressed reasoning in most outputs.
The pinned VLMEvalKit raw exact
matching reports 513/765 = 67.06%; its 26 disagreements all reject duplicated
first-line/final explicit option letters as `Z`. Original xai packaging was
479/765 = 62.61% (AGS 478/765). See `REPLICATION.md`; do not mix these protocols.

The original first-128 development references were 91/128 for directive-dominated
selection and 87/128 for reasoned-v2 under the preceding parser. Re-score the same
IDs uniformly when assessing the new v4 prompts; those development results are
not a held-out final benchmark.

With the aligned VLMEvalKit first-16 packaging, NF4 direct is 9/16 and reasoned
CoT is 10/16. V7 online and online-CF are both 7/16 with identical answers; CF
executes 20 interventions but raises uncertainty without correcting an answer.
V9 prevented unverified final synthesis from overwriting the initial candidate
when no later fact was verified. V10 added reference-blind option-consistency
scoring (historically named `option_counterfactual` in code). Current outcomes
are reported above. The V7/control input-pixel mismatch prevents causal attribution.

## Historical run provenance (relative to /data_cinema/zqc/CVPR)

- `runtime/queues/reasoned_bf16`: the 765-case reasoned-v2 run removed the embedded
  direct-answer directive and completed at stored-parser 522/765 (AGS 523/765).
  The paired FaithAct job reused that hashed output and is now complete.
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

Finish the running full-765 layout-v5 CoT/FaithAct sequence, V15 CF, and queued
V16 NF4 development comparison. The subsequent BF16 V16 online/CF run will use
the same IDs 128–255 as the completed CoT validation slice. The existing FaithAct
gap and V15's within-run regression remain unresolved. Keep the declared prompt,
parser, and method settings fixed while these comparisons complete; do not
extrapolate the 75% validation score to the full benchmark or to our method.

Completed v3 raw traces, code snapshots and corrected report were published to
HF `trainingfree/milestones/20260922_pilot4_v3`, commit
`e0db220bee7f1527965cbf2963e77363dbf4162c`. GitHub code publication is batched at
this milestone. Preserve other-computer changes. No current score justifies a superiority
claim. `paper/positioning.md` notes that FaithAct's paper already discusses
online verification, unlike the structure of its released Qwen script.
