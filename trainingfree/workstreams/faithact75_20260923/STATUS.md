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

An NF4 16-case diagnostic is running from the isolated snapshot. V22 is not an
accuracy improvement until the completed paired output says so; CF remains
deferred until the non-CF diagnostic is worth the added cost.

## Active server paths

- Workstream root: `/data_cinema/zqc/CVPR/workstreams/faithact75_20260923`
- BF16 validation queue: `queues/paper_literal_faithact_validation_bf16`
- BF16 validation run: `runs/paper_literal_faithact_validation128_bf16`
- V22 queue: `queues/v22_blind_read_nf4`
- V22 run: `runs/v22_blind_read_nf4_pilot16`
- Development selector result: `analysis/selective_faithact_v1_dev128.jsonl`

