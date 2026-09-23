# FaithAct replication and Evidence-First continuation — 2026-09-23

## Fixed BF16 validation, RealWorldQA IDs 128–255

All rows use Qwen2.5-VL-7B-Instruct, the same VLMEvalKit case images,
`max_pixels=1003520`, and reference-blind extraction. The paper-literal
`Think step by step.` run was frozen before these IDs were scored. It reuses
its exact 128 initial chains for the released-code FaithAct rewrite.

| Output | v4 | v5 formatting replay | Raw AGS | Raw VLMEvalKit |
|---|---:|---:|---:|---:|
| Paper-literal CoT | 81/128 | 83/128 | 36/128 | 53/128 |
| Released-code FaithAct | 77/128 | 77/128 | 23/128 | 59/128 |
| Frozen selective FaithAct v1 | 91/128 | 91/128 | n/a | n/a |
| Independent layout-v5 CoT | 96/128 | 96/128 | 96/128 | 92/128 |

The v5 changes are two recovered terminal option lines (IDs 150 and 185) in
the *unchanged* paper-literal CoT output, not model gains. Replay across six
files (3,316 rows) changed no already-valid v4 label and changed no full-765
layout-v5/FaithAct label. The frozen selector gains 9 valid corrections and
causes 1 regression versus the v5 paper-literal anchor, but remains five
answers below layout-v5. Its earlier development 97/128 (75.78%) was not
confirmed on validation. A simple three-prompt majority diagnostic on all
765 cases scores 535/765, below layout-v5's 542/765. Applying the same
unanimity rule to the full layout-v5/released-rewrite cache, with the two
available full support runs (directive and reasoned-v2), scores 546/765
(71.37%) under uniform parser v5, also below the target of at least 574/765
for 75%. This full-cache diagnostic has a different anchor/support combination
from the frozen paper-literal validation selector, so it is not a held-out
confirmation of that selector.

FaithAct changes 12 wrong CoT answers to correct and 16 correct ones to wrong
under v4 on this validation slice (net -4, exact McNemar p=0.572). Under v5,
two apparent corrections are correctly reclassified as both methods already
right, so the true paired result is 10 corrections and 16 regressions (net -6).
The actual tool path ran on 127 cases, extracted 759 noun records, and supplied
boxes to 112 final prompts. Of those noun records, 468 had existence score
below 0.4. This coverage rules out “no tool calls” as the explanation for the
negative result. Changed-answer traces show unsupported object counts,
left/right or relative-depth reversals, numeric distance guesses from image
boxes, and eight rewritten answers without a valid option label. There are
also genuine evaluator disagreements: AGS's regex and VLMEvalKit's raw
matching often reject long-form final answers; canonical-final scoring is
only the same extracted answer fed to VLMEvalKit, not independent accuracy.
Specifically, AGS can stop at a colon before a newline-delimited answer,
fail to reparse an option letter after stripping `is`, or count an uncommitted
option mention by reference-conditioned substring fallback. VLMEvalKit raw
matching likewise rejects many long explanations even when a unique final
label is recoverable. These are format-protocol differences, not evidence that
any particular method got the visual question right.

## Replication limit and interpretation

The public FaithAct release contains noun extraction, CLIP/Poll,
GroundingDINO, and the whole-chain rewrite. It does not contain the Qwen
initial-generation adapter used for its Table-2 RealWorldQA result or that
task-accuracy evaluator. Its runnable Qwen script consumes a precomputed chain
and is configured for POPE. The released GCoT template, the paper-literal
prompt, and layout-v5 produce markedly different initial scores, so none may
be silently named the original 70.1% CoT protocol. The paper-appendix rewrite
prompt and detector thresholds also differ from the released script. Exact
74.5% parity is therefore unestablished; this is a declared reconstruction,
not a refutation of the paper result.

## Evidence-First controller

V25's four NF4 engineering cases scored 1/4 and exposed a control bug:
the schema decoder sometimes returned the *string* `"null"` as a final
answer, which could be marked `verified_by_model`. The parser now maps quoted
null/none sentinels to an actual null; an isolated V26 snapshot uses this fix.
V26 also changes the unverified safety anchor from paper-literal CoT to the
stronger layout-v5 CoT while retaining option-blind, candidate-blind visual
acquisition, audit-before-commit, at most three repairs, explicit uncertainty,
and forward continuation. The CoT chain is not promoted to verified premises.
V26's paired NF4 16-case run is active; its score is not reported until both
arms complete. The older V20 BF16 run was still incomplete at 83/128 cases
when its partial diagnostic was taken: 54/83, versus its own valid
paper-literal anchor 51/81 on the 81 anchor-answered cases (two corrections,
zero regressions). The other two initial answers were unavailable. A nearly
simultaneous 84-case snapshot was 55/84 versus layout-v5 64/84. These changing
partial counts are diagnostics, not final method scores; the own-anchor gains
do not erase the weaker paper-literal starting point. An exploratory
arbitration of layout-v5/directive disagreements using V16's
`verified_by_model` status changed three validation answers, one correction
and one regression, leaving layout-v5 at 96/128. Current model-verifier
agreement is therefore not a reliable selector by itself.

## Reproducible artifacts

All original outputs and audits remain under
`/data_cinema/zqc/CVPR/workstreams/faithact75_20260923/`:

- `runs/paper_literal_faithact_validation128_bf16/{cot,faithact}.jsonl`
- `analysis/paper_literal_faithact_validation128_diagnostics.json`
- `analysis/scorer_suite_v5_{cot,faithact}_validation128/summary.json`
- `analysis/parser_v5_replay_six_runs.json`
- `analysis/selective_faithact_v1_parser_v5_validation128.summary.json`
- `analysis/selective_faithact_v1_parser_v5_layout_full765.summary.json`
- `analysis/majority_v5_layout_directive_reasoned_full765.summary.json`
- `snapshot_v26/` and `runs/v26_layout_anchor_nf4_pilot16/`
- `queues/v26_layout_anchor_bf16_validation/` (queued for free memory)

These paths are server artifacts, not Git-stored dataset copies. No 75%+
full-benchmark method claim is supported by current measurements.
