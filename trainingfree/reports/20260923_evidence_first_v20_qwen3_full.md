# Evidence-First validation and Qwen3 controls — 2026-09-23

## Same-dataset results, with precision and protocol separated

All accuracy counts below cover the public VLMEvalKit-derived RealWorldQA
cases and use the same reference-blind committed-answer v5 extractor. They do
not share a model/input protocol unless stated.

| Run | Precision | IDs | Correct | Scope |
| --- | --- | --- | ---: | --- |
| Qwen2.5-VL-7B layout-v5 CoT | BF16 | 0–764 | 542/765 (70.85%) | strongest completed single Qwen2.5 full baseline |
| Qwen2.5-VL-7B layout-v5 CoT | BF16 | 128–255 | 96/128 (75.00%) | fixed validation slice |
| Qwen2.5-VL-7B paper-literal CoT | BF16 | 128–255 | 83/128 (64.84%) | V20's prompt-family starting point |
| FaithAct released rewrite | BF16 | 128–255 | 77/128 (60.16%) | separate reconstructed baseline |
| Evidence-First V20 | BF16 | 128–255 | 87/128 (67.97%) | stepwise audit, repair, uncertainty; no spatial CF |
| Qwen3-VL-8B layout-v5 control | NF4 | 0–764 | 524/765 (68.50%) | different frozen backbone, 1M-pixel input |
| Qwen2.5 layout/directive + Qwen3 three-vote | mixed BF16/NF4 | 0–764 | 555/765 (72.55%) | answer ensemble, **not** the stepwise paper method |
| Five-vote including Qwen2 reasoned/FaithAct | mixed BF16/NF4 | 0–764 | 541/765 (70.72%) | exploratory control; worse than three-vote |

V20's 128 cases have 125 extractable initial answers. Among these exact paired
cases, the initial/final scores are 83/125 and 85/125: two initial errors were
corrected (IDs 199 and 201), with no correct initial answer overturned. Three
initial answers are unresolved by the shared extractor; V20 got two of those
right at finalization. This explains the 83/128 to 87/128 aggregate rise; it
does **not** establish a four-case correction gain among valid initial answers.
Against the independent layout-v5 CoT on the same IDs, V20 has five gains and
14 losses (87 versus 96). V20 made 576 recorded audits and 221 repair attempts,
ended uncertain on 72/128 cases, and averaged 14.79 VLM calls and 137.08 s per
case. No spatial counterfactual was executed in this V20 arm. It is not the
final counterfactual method and cannot substantiate a 75%+ claim.

The 765-row Qwen3 control was generated in two disjoint, resumable shards
(0–31 and 32–764), merged only after checking exact ID coverage, uniqueness
and public references. Its NF4 precision, Qwen3 architecture and image
processor differ from the Qwen2.5 BF16 runs, so the 524/765 score is a
separate backbone control, not a method delta. The three-vote rule received
only extracted answer labels; it did not see references at prediction time.
Its 555/765 score is 19 correct answers short of the 574/765 required for at
least 75%. An exploratory five-vote rule fell to 541/765, showing that more
weakly correlated votes are not automatically better.

## Failure mechanism observed in step-boundary pilots

V26's eight NF4 engineering rows changed no answers and verified no visual
facts. V27 allowed visual support down to 0.65 only for an exact candidate-blind
readback; its five saved rows also changed no answers (2/5 versus 2/5 anchor).
V28 used the Yes-over-No boundary 0.5 under the same exact-blind condition;
its first five rows remained 2/5, but ID 0 changed from a wrong B to a wrong A
and marked an incorrect leftward wheel claim verified. These are engineering
pilots, not BF16 benchmark comparisons.

V29 replaced only the neutral blind reread with a separate frozen Qwen3 model.
It repeated the ID-0 false claim: Qwen3 answered C under its original layout-v5
task prompt, but said *left* when asked the neutral evidence question on the
full image and magnified region. A focused probe reproduced *left* on full-only,
full+crop and crop-only neutral views; adding original task options to the
observation query changed its wording to *straight*. This direct trace shows
that model independence, crop magnification and text-only readback agreement
do not guarantee a correct visual fact. The V29 two-case run was stopped with
its traces preserved; no full result is claimed.

## Next controlled step

The Qwen3 official RealWorldQA runner uses a larger image budget than our
1M-pixel control. An independent Qwen3 NF4 layout-v5 high-resolution run on
the fixed IDs 128–255 is queued with 602,112 minimum and 4,014,080 maximum
pixels, keeping decoding, model, prompt and extractor unchanged. The existing
1M-pixel Qwen3 result on those same IDs is 85/128, so a high-resolution gain
must be measured, not assumed. The next method variant must test cross-context
consistency before any fact is marked verified; otherwise V28/V29's false
verification mechanism persists. If it conflicts after at most three repairs,
the point remains uncertain and reasoning continues, as specified by the
research contract.

No complete RealWorldQA Evidence-First result ≥75% is established.

## Artifacts

- V20 BF16 raw: `/data_cinema/zqc/CVPR/runs/realworldqa_vlmevalkit_7b_bf16_v20_evidence_first_validation128/evidence_first.jsonl`
- V20 diagnostic: `/data_cinema/zqc/CVPR/workstreams/faithact75_20260923/analysis/v20_bf16_validation128_diagnostics.json` (raw SHA256 `0ee23a8da61f3cd51d7bbbb17ae7946865d8812da7bc1f324330daed36c389dc`)
- Qwen3 merged 765: `/data_cinema/zqc/CVPR/workstreams/faithact75_20260923/analysis/qwen3_nf4_layout_v5_full765.jsonl`
- Three-vote and five-vote outputs and source hashes: `/data_cinema/zqc/CVPR/workstreams/faithact75_20260923/analysis/majority_qwen25_layout_directive_qwen3_nf4_full765.*` and `majority_five_existing_full765.*`
- V27/V28/V29 pilots and blind-view probe: `runs/v27_exact_blind_nf4_pilot16/`, `runs/v28_blind_majority_nf4_pilot16/`, `runs/v29_dual_blind_nf4_pilot16_slow/`, `analysis/blind_view_probe_case0.json` under the same workstream root.
