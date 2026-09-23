# Prompt v6 validation and answer-guard regression (2026-09-23)

## Scope and protocol

- Dataset: RealWorldQA 765 cases, fixed development IDs 0–127 and validation IDs 128–255.
- Frozen backbone: Qwen2.5-VL-7B-Instruct. Model inference used the same 1,003,520 maximum pixel budget and 512 output-token cap as the layout-v5 control.
- Answer extraction: the same committed reference-blind parser, not a post-hoc answer-key edit.
- Prompt v6 was designed on the NF4 development slice. BF16 validation was launched only after the prompt was frozen.

## Measured results

| Precision and slice | Layout-v5 | Discriminative-evidence-v6 | Paired outcome |
| --- | ---: | ---: | --- |
| NF4, development 0–127 | 85/128 | 92/128 | 11 gains, 4 losses |
| BF16, validation 128–255 | 96/128 | 94/128 | 6 gains, 8 losses |

The validation result does **not** establish an improvement. V6 was not expanded to the full 765 cases. Its exact FaithAct-style `Evidence:` blank-line/numbered layout occurred in 74/128 NF4 development outputs, compared with 120/128 for layout-v5, so it is also a poor drop-in replacement for FaithAct's released blank-line splitter. A format-preserving v7 is a separate development candidate.

V7's matched NF4 development result was 89/128, versus v5 85/128 and v6 92/128. Its exact evidence layout was 121/128. BF16 validation on IDs 128–255 finished at 93/128, versus v5 96/128 and v6 94/128. V7 gained four and lost seven relative to v5; all 128 BF16 outputs followed the exact `Evidence:\n\n1.` structure. Thus formatting was fixed, but accuracy was not. V7 was not expanded to 765.

## Option-likelihood deduction diagnostic

A frozen Qwen2.5 model scored all answer labels in one common-prefix forward pass without receiving the candidate answer or reference. The standalone NF4 development score was 91/128. A margin threshold of 0.75, fixed after development, changed eight v6 answers and gave 97/128 (six corrections, one regression, plus one correctness-neutral change). This was a diagnostic of a possible deduction gate, **not** a stepwise-method score.

On the BF16 validation slice, standalone option likelihood scored 90/128. Applying the already fixed 0.75 threshold to the stronger layout-v5 anchor produced 94/128 versus the anchor's 96/128 (one correction, three regressions). Applying it to v6 produced 94/128 versus v6's 94/128 (two corrections, two regressions). Thus the development 97/128 did not validate, and this gate is not promoted to the Evidence-First controller. The selector's code never reads the reference for its decision; correctness is computed only afterward.

Raw outputs: server `workstreams/faithact75_20260923/runs/qwen25_nf4_prompt_pair_dev128_v5`, `..._v6`, and `runs/qwen25_bf16_discriminative_v6_validation128`. The BF16 validation queue completed without errors. The 14 discordant validation IDs are recorded in the accompanying analysis and can be regenerated from those two prediction files; no reference labels were passed into either model prompt.

## Evidence-First integrity regression

The controller had compared a raw final-answer string with option letters. `A. Left` is a valid A under the final answer parser but previously skipped option-consistency and the two-independent-verified-facts changed-answer guard. The comparison now calls the same reference-blind option extractor as scoring. Unit tests cover formatted labels and quoted `null`; 52 focused tests pass.

The isolated NF4 ID 0 regression with the patch confirmed `answer_withheld=true` and `verified_visual_n=1` for the proposed `A. Left` against the tentative B anchor. Its final B was still wrong against reference C. Candidate-blind and crop-based verification had also accepted a false leftward wheel observation; this remains a hard verifier failure, not a correction. Repeated acquisition of the same point created 26 model calls and an uncertain fallback. The core controller now receives exact repetitions for its targeted bounded redirection instead of generic parse retries; this follow-up optimization has unit coverage but no measured accuracy claim.

## Decision

Continue the per-step acquisition → audit → at most three repairs → explicit uncertainty → next step architecture. Do not treat self-reported verifier acceptance as truth. A new answer-only-anchor controller removes the upfront CoT fallback from the actual stepwise path; it is queued for an NF4 engineering pilot. V6/V7 accuracy and V30 integrity are separate findings; none is a full-765 Evidence-First ≥75% result.
