# Status: 2026-09-23

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
