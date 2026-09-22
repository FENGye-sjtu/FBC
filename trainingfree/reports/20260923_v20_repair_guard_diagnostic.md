# V20 paired NF4 diagnosis and V21 repair guard

These are first-16 RealWorldQA/VLMEvalKit engineering results, not BF16 paper
replication or a held-out benchmark gain. Both arms use Qwen2.5-VL-7B NF4,
1003520 max pixels, 512 new tokens, the same 16 case IDs and dataset SHA256
`c873cd0f9e61f937834ad6e81964484d165ea6fb72243e788a2fdc26e0b8ad87`.
The reference-blind `committed_answer_v4` parser re-scored all 16 public cases.

| V20 arm | Initial paper-literal CoT | Final | Corrections / regressions | Status | Repeated-step events |
| --- | ---: | ---: | ---: | --- | ---: |
| Evidence First | 10/16 | 10/16 | 0 / 0 | 4 model-verified, 12 uncertain | 44 |
| Evidence First + image CF | 10/16 | 10/16 | 0 / 0 | 2 model-verified, 14 uncertain | 43 |

The CF arm performed 11 target masks and 22 matched control masks without
changing any final answer. Its mean latency was 120.84s versus 100.94s for the
non-CF arm on shared hardware; this is not a controlled throughput benchmark.
The existing matched direct selection scored 9/16; the 10/16 result is already
present in V20's CoT anchor, not an Evidence First correction. The paper-literal
prompt's separate 94/128 development result and the FaithAct prompt/protocol
audit are in `20260922_faithact_prompt_protocol_audit.md`.

Original server files (preserve them):

- `/data_cinema/zqc/CVPR/runs/realworldqa_vlmevalkit_7b_nf4_pilot16_v20_evidence_first/evidence_first.jsonl`, SHA256 `d622f7bc0c6595b00c73c142b2857e3c96804995bfe43b29f0cf9defa6b7255d`.
- `/data_cinema/zqc/CVPR/runs/realworldqa_vlmevalkit_7b_nf4_pilot16_v20_evidence_first/evidence_first_cf.jsonl`, SHA256 `361ec2dbaed4feb38f0f6332ce4e7050bd514fcf1e1b9e6d511c12829c69fffc`.

The early V20 trace persisted `tentative_direct_anchor`, although protocol
`evidence_first_v7_paper_literal_anchor` and its saved `anchor_raw` show a CoT
generation. The analyzer now reports the actual CoT provenance while preserving
the recorded field and raw output. It changes no answers or scores.

Case 1 illustrates a protocol bug: after two accepted visual observations,
the repair path emitted a deduction with an empty evidence question. The
controller forced its kind to `visual`, accepted a previously verified bar
count again, and later exhausted repeated proposals. V21 rejects deductions
in visual repair and exact repeated visual claims even with a changed evidence
question. This is an integrity fix, not an accuracy result; paraphrase-level
duplicates and correlated verifier errors still require measurement.

V21 runs from the separate snapshot
`/data_cinema/zqc/CVPR/code_snapshots/v21_repair_guard` on GPU 2, with original
BF16 worker code and queues untouched. Snapshot `evidence_first.py` SHA256:
`7d23924c9907bd125366c24378ac83521c63c15fb05edc78429bcafba6ece916`.
Its output is
`/data_cinema/zqc/CVPR/runs/realworldqa_vlmevalkit_7b_nf4_pilot16_v21_repair_guard`;
score only after all 16 records exist in each arm, compare paired corrections
and regressions against the same V20 IDs, and record any model/runtime errors.
