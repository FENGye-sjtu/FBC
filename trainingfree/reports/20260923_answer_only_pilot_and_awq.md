# Answer-only stepwise pilot and larger frozen-backbone preparation (2026-09-23)

## V32 Evidence-First protocol

V32 removes the earlier full-CoT tentative fallback. Before the first audit it asks the frozen Qwen2.5-VL-7B model only for a concise answer, labels that answer uncertain, and never gives it to the value-blind evidence-acquisition prompt. It then follows the existing one-visual-step → audit → up to three repairs → explicit uncertainty → next-step controller. A conclusion may use only verified ledger entries. This closes a narrative weakness in earlier engineering variants that generated a complete tentative chain before entering the stepwise controller.

The NF4 engineering pilot covered RealWorldQA IDs 0–7. It finished **6/8**, exactly the same as its own answer-only anchor **6/8**, with **zero changed answers**, six `answered_with_uncertainty` outcomes, two `verified_by_model` outcomes, and 11.5 model calls/case on average. This is a functional smoke test, not a paper accuracy result. ID 0 showed bounded repair and forward continuation, but repeated neutral rereads stayed on the wrong wheel interpretation; the controller did not correct it. No 128- or 765-case V32 run is justified by these eight rows.

Raw run: `/data_cinema/zqc/CVPR/workstreams/faithact75_20260923/runs/v32_answer_only_stepwise_nf4_pilot8`. The first snapshot's `finalization.source` string inherited the parent label `tentative_paper_literal_cot_anchor` even though `anchor_raw` is an answer-only response. The code now corrects that provenance label for subsequent runs; the immutable pilot rows are not rewritten. The true protocol is visible in `controller_protocol=evidence_first_v32_answer_only_anchor`, its code snapshot, and the short `anchor_raw` fields.

## Frozen 32B control / independent verifier candidate

The [official Qwen2.5-VL-32B-Instruct-AWQ release](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct-AWQ/tree/main) is a 20.7 GB checkpoint. The pinned public revision `66c370b` was downloaded only to `/data_cinema/zqc/CVPR/models/Qwen2.5-VL-32B-Instruct-AWQ`; on-disk model size is about 20 GB. AutoAWQ 0.2.9 and zstandard were installed only in `/data_cinema/zqc/CVPR/vendor/autoawq`; the pre-existing VBC environment was not changed. The downloaded safetensors headers indicate 19.28 GiB of stored tensor data, which the new explicit placement logic counts by actual dtype rather than pretending all AWQ packed tensors are BF16. The loading/8-case baseline smoke is queued for a safe multi-GPU memory window; **no 32B score exists yet**.

Any stronger-backbone score will be reported separately from a same-backbone Evidence-First gain. If the 32B model proves useful as an independent verifier, its extra inference cost and precision will be measured explicitly; neither model independence nor model scale is assumed to guarantee true evidence.
