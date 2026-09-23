# Status: 2026-09-23 (active continuation)

## Latest measured and queued work

The full continuation report is `../../reports/20260923_evidence_first_v20_qwen3_full.md`.
V20 BF16 completed 87/128 versus its own paper-literal 83/128 starting point
but below the stronger independent layout-v5 96/128. Qwen3 NF4 layout-v5
completed 524/765. Qwen2.5 layout/directive plus Qwen3 majority completed
555/765 (72.55%), still 19 correct answers below the 75% threshold. The
same-model Qwen3 high-resolution validation completed 86/128 versus 85/128
at low resolution, and did not improve the 95/128 three-vote result on that
slice. It was not expanded to 765.

Qwen2.5 NF4 prompt development IDs 0–127: discriminative evidence v6 reached
92/128 versus layout-v5 85/128, with 11 paired gains and four losses. This is
a development result, not a BF16 or full-765 result. V6 relaxed the exact
FaithAct blank-line layout: only 74/128 outputs match `Evidence:\n\n1.`
versus 120/128 for v5. A layout-preserving v7 is running for a matched NF4
development comparison. BF16 v6 completed 94/128 on fixed IDs 128–255,
below the 96/128 BF16 layout-v5 anchor (six gains, eight losses), so a full
v6 run is not justified. V7 then reached 89/128 on NF4 development while
restoring the strict evidence layout in 121/128 outputs; BF16 validation
reached only 93/128 versus v5 96/128, despite exact layout on all 128 cases.
A frozen option-likelihood gate reached 97/128 only on NF4
development, but its fixed 0.75-margin rule reduced the stronger BF16
layout-v5 validation anchor from 96/128 to 94/128. It is not promoted to the
method. See `../../reports/20260923_prompt_v6_validation_and_guard.md`.
The answer-only-anchor V32 stepwise NF4 pilot completed 6/8, equal to its own
6/8 anchor, with no changed answers, six uncertain outcomes and 11.5 calls/case.
It does not create an entire tentative CoT before its first audit. The first
snapshot mislabeled the fallback source as paper-literal; the actual prompt
and protocol were answer-only, and subsequent code corrects the metadata.
See `../../reports/20260923_answer_only_pilot_and_awq.md`.

An official frozen Qwen2.5-VL-32B-Instruct-AWQ checkpoint (20.7 GB) is now
fully downloaded inside `/data_cinema/zqc/CVPR/models`, with AutoAWQ installed
only in the CVPR vendor tree. Its 8-case loading/accuracy smoke is queued for
sufficient combined free GPU memory. This is a separate backbone/verifier
control; it has no score yet and cannot be counted as an Evidence-First gain.

V30 fixes a controller formatting bug: `A. Left` now normalizes to option A
before the changed-answer guard. A one-case NF4 regression test confirmed the
two-independent-verified-facts gate engages. It also exposed repeated attempts
to reacquire the same fact; the core controller now receives those repetitions
for targeted redirection instead of treating them as generic parse errors.
This case remained wrong and uncertain, so the guard is an integrity fix, not
an accuracy gain.

- V26's first eight NF4 rows were 4/8 versus the same layout-v5 anchor's
  4/8, with no answer changes, no verified visual facts, and 8/8 uncertain
  fallback answers. The paired BF16 V26 queue was cancelled before launch.
  Both obsolete V26 processes were terminated after preserving their traces.
- V27 changes only the *visual* audit: a sub-0.8 claim can pass at 0.65 if a
  candidate-blind reread yields exactly the same atomic fact and both blind
  consistency and original-image support meet 0.65. Its five completed NF4
  rows were 2/5, exactly the 2/5 own anchor, with no changed answers. Four
  visual facts passed; the run was stopped to release GPU memory. These
  thresholds are engineering heuristics, not calibrated correctness.
- V28 tests the natural Yes-over-No boundary (0.5) only for the same exact
  candidate-blind visual agreement. Deductions still require 0.8; all visual
  candidates still pass an audit before entering the ledger, followed by at
  most three repairs, explicit uncertainty and forward continuation. Its
  first five NF4 rows were 2/5 versus 2/5 layout anchor, including a wrong
  B->wrong A change and an incorrectly "verified" leftward wheel claim on ID
  0 (reference C/right). The run was stopped after preserving five rows.
- A frozen Qwen3-VL-8B-Instruct NF4 layout-v5 32-case **backbone control**
  finished 23/32, versus 22/32 for the existing Qwen2.5 BF16 layout output on
  the same IDs. A three-answer Qwen3/layout/directive vote was 25/32 only on
  this development slice, not a full-benchmark or method result. The remaining
  733 Qwen3 predictions subsequently completed; the merged 765-case score is
  524/765. Qwen3 is not comparable as a Qwen2.5 method gain.
- V29 routes only the candidate-blind visual reread to a separate frozen
  Qwen3 model. Its first two NF4 rows made no net gain; ID 0 still admitted
  the wrong leftward wheel claim despite the independent model's correct C
  answer under the original layout-v5 question. It was stopped after two
  rows. A focused view/prompt probe showed Qwen3 said *left* under full-image
  neutral read, full+crop neutral read, and crop-only neutral read; it said
  *straight* when the original options were added to the observation query.
  Thus model independence and localized views do not themselves establish
  visual truth. The correct C arose only from Qwen3's original task-answer
  generation, so future gates must detect cross-context inconsistency.
- Two independent Qwen3 jobs briefly attempted to load on the same GPU from
  separate memory queues and both failed with CUDA OOM. No benchmark rows or
  unrelated processes were modified; the 733-case run is resumable and has
  since completed alone. Single-GPU launchers now require sufficient memory
  on *one* card rather than sum two cards they do not actually use.
- The older V20 BF16 128-case validation is complete at 87/128. It uses a
  paper-literal CoT fallback rather than layout-v5 and thus cannot be compared
  to the 96/128 layout-v5 validation baseline as a same-anchor method delta.

No completed full-765 Evidence-First run has reached 75%+.

Fixed validation and parser-v5 replay are summarized in
`../../reports/20260923_faithact_validation_and_v26.md`. On IDs 128–255,
paper-literal CoT/FaithAct score 83/77 under v5; the frozen selector is 91/128,
while layout-v5 alone is 96/128. The selector's development 75.78% did not
validate. Full-765 layout-v5/FaithAct remain 542/510; the uniform-v5
selective diagnostic is only 546/765. V25 exposed quoted-`null` final-answer
laundering, now fixed. No completed full Evidence-First score is 75%+.

## FaithAct reproduction conclusion

The upstream release is sufficient to reproduce the public noun extraction,
GroundingDINO/CLIP polling, confidence fusion and whole-chain rewrite. It is not
sufficient for an exact RealWorldQA Table-2 reproduction because it omits the
Qwen initial-generation adapter and the task-accuracy evaluator. Its runnable
Qwen entry point is configured for POPE and consumes precomputed chains.

Our controlled reconstruction keeps these gaps explicit:

- released `TEMPLATE_GCOT` + released rewrite: 78/128 -> 78/128;
- paper-literal `Think step by step.` + released rewrite: 94/128 -> 87/128;
- layout-v5 full reconstruction + released rewrite: 542/765 -> 510/765 under
  parser v4.

On the paper-literal development pair, FaithAct makes 12 corrections and 19
regressions. Six rewritten answers are invalid labels or unresolved prose. The
main semantic failures are unsupported count changes, metric estimates inferred
from pixel boxes, orientation changes without discriminating cues, and traffic
decisions derived from object presence alone. Raw AGS/VLMEvalKit scores also
under-count long-form answers, so all model comparisons use the same
reference-blind committed parser and report raw evaluator compatibility only as
a separate diagnostic.

## Frozen selective candidate

`selective_faithact_v1_all_supports` retains the paper-literal chain unless a
valid changed FaithAct label is also predicted by both the directive and
layout-v5 support prompts. Selection does not receive the reference answer.

Development result: 97/128 = 75.78%, versus anchor 94/128. It accepts 14
revisions: eight corrections, five regressions and one correctness-neutral
change. This is a prompt-development result, not a held-out 75%+ claim.

The fixed validation task covers IDs 128-255 in BF16. It will generate the
paper-literal/FaithAct pair and then apply the frozen selector using existing
same-precision directive and layout-v5 outputs. The queue and outputs are under
`/data_cinema/zqc/CVPR/workstreams/faithact75_20260923`; no earlier run is
modified.

## Evidence-First candidate

V20's first-16 result was 10/16, identical to its paper-literal anchor, with no
correction. Failure traces show that answer-like conclusions such as “uphill”
or “same direction” can be proposed as visual facts and then confirmed by the
same model. Several deductions also reach the 300-character schema limit.

V22 adds a candidate-blind visual reread before the existing support audit. The
reread receives the neutral evidence question and image/crop but not the
candidate claim. A separate comparison must show that the reread entails the
claim. Obvious causal/conclusion language is rejected from visual claims, and
deductions are capped at 180 characters. The original three-repair limit,
explicit uncertainty and forward-only ledger remain unchanged.

V21 completed its non-CF 16-case guard run at 10/16, exactly its 10/16 anchor,
with zero corrections and zero regressions. It produced 181 rejected/invalid
proposal events, so the integrity guard did not improve accuracy and added
substantial control churn.

V22's first case showed that the candidate-blind read can strongly reject an
unsupported candidate, but its lexical inference-language gate then rejected
the corrected reread wording repeatedly. The case retained the wrong anchor and
took 551 seconds. That diagnostic was stopped after one preserved row rather
than spending the remaining 15 cases on a known control-flow defect.

V23 keeps the candidate-blind semantic audit and removes the brittle lexical
gate. Its first two attempts still ignored the contradictory blind observation,
so the partial run was stopped with its events preserved.

V24 promotes a resolved blind observation only as a new candidate; it must pass
another blind comparison and the original support gate before entering the
ledger. The first 3B smoke exposed a dependency on a newer private base-class
helper and failed before producing a score. V24b made that duplicate check local
and exercised the intended loop: support rose from 0.59 to 0.68 after blind
repair but stayed below 0.8, so the fact remained unverified. The smoke was
stopped after preserving ten events because it then repeated an equivalent
blind observation.

V25 suppresses promotion when the blind observation is equivalent to the
candidate already audited. Its separate 7B four-case NF4 queue is waiting for
safe spare memory. CF remains deferred until the non-CF diagnostic is worth the
added cost.

The same frozen unanimity gate was also tested on the existing BF16 online
validation output. It reduced the 96/128 layout anchor to 92/128 (two corrections
and six regressions). Therefore unanimity is not treated as a generic selector;
the 97/128 development result still requires the like-for-like FaithAct
validation below.

## Active server paths

- Workstream root: `/data_cinema/zqc/CVPR/workstreams/faithact75_20260923`
- BF16 validation queue: `queues/paper_literal_faithact_validation_bf16`
- BF16 validation run: `runs/paper_literal_faithact_validation128_bf16`
- V22 queue: `queues/v22_blind_read_nf4`
- V22 run: `runs/v22_blind_read_nf4_pilot16`
- V23 queue: `queues/v23_blind_semantic_nf4`
- V23 run: `runs/v23_blind_semantic_nf4_pilot4`
- V25 queue: `queues/v25_blind_repair_nf4`
- V25 run: `runs/v25_blind_repair_nf4_pilot4`
- Development selector result: `analysis/selective_faithact_v1_dev128.jsonl`
