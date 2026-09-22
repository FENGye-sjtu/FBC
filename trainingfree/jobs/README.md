# Experiment requests

These JSON files are versioned requests, not instructions to launch every file.
Older requests are retained for provenance; some intentionally reproduce fixed
failures. Inspect STATUS.md and the server worker result before submitting.

Current sequence on 2026-09-22:

Update after the historical list below was written: full layout-v5 and the
released-GCoT/released-FaithAct 128-pair are complete; see `STATUS.md` for the
measured scores. Parser v4 adds only the audited Markdown-bold final-marker
recovery on top of v3. The active BF16 queue is job 003b (paper-appendix
FaithAct), followed by 003c, V16 validation and V18 evidence-first validation.
The NF4 queue finishes the old V17 CF trace before the new V18 16-case gate.
Later bullets mentioning jobs still running/no result are retained as historical
launch-time state, not current status.

- The old BF16 sequence is complete: reasoned-v2 FaithAct plus the three 128-case
  prompt comparisons. Brief-v4 is 89/128 correct with 34 inspectable outputs;
  adaptive-v4 is 80/128 with 121 valid labels and 69 inspectable outputs;
  layout-v5 is 90/128 (70.31%) with 128 valid labels and 128 inspectable outputs.
  The same-parser direct reference is 92/128 and old reasoned-v2 is 87/128.
  Layout-v5 had zero literal placeholders, with actual visual statements in
  inspected samples. It is the selected coverage/accuracy candidate; 70.31%
  is its development score, not a full-765 result.
- The frozen parser is `committed_answer_v3`, SHA256
  `49c5a365e342cf423be14fb4ef9d59e49e1bcf4f98f1036f101f0a71c1140a83`.
  Its additive fallback only extends unresolved v2 outputs: an audit of 20 files
  and 3,794 rows changed zero already-valid v2 labels. It retains v2's semantic
  limitations. The full-765 scores remain 541/526/528, and prompt-development
  scores remain 89/80/90. All reporting tools use the shared raw-source rule and
  record parser version. Current scorer reports use `20260922_parser_v3_*`.
- BF16 job 001, `7b_bf16_layout_v5_faithact_dev128_gpu.json`, is complete at
  88/128 (68.75%), versus its cached CoT 90/128: 10 corrections, 12 regressions,
  McNemar p=0.8318. All 128 cases used tools; 85 have bounding-box evidence.
  Official AGS raw is 23/128 (17.97%), VLMEvalKit raw is 68/128 (53.125%), and
  canonical final is 88/128. Preserve these raw scores. Prose/option-label
  contradictions mean correct labels do not establish faithful reasoning.
- BF16 job 002, `7b_bf16_layout_v5_validation128.json`, has completed IDs
  128–255 at 96/128 (75.00%), with 127 inspectable outputs. AGS raw is also
  96/128; VLMEvalKit raw is 92/128, canonical final 96/128. Same-ID old direct
  is 91/128 and reasoned-v2 is 89/128. Paired wins/losses are 11/6 versus direct
  (p=0.3323) and 11/4 versus reasoned-v2 (p=0.1185). These are separate validation
  results, not statistically significant or full-benchmark improvements.
- `runtime/queues/layout_v5_bf16` is now running job 003,
  `7b_bf16_layout_v5_full_replication_gpu.json`, with full-765 CoT followed by
  FaithAct. No full-layout-v5 result is complete. Job 004,
  `7b_bf16_v16_online_validation128.json`, is queued afterward for online and CF
  on IDs 128–255 in BF16, not NF4. This online validation job has not started and
  has no score yet.
- This BF16 launcher enables a separate tool GPU through
  `--tool-gpu-free-mib`; both model and tool allocations must fit before launch.
  At the current launch, physical GPUs 1/0 hold weights and physical GPU 3 holds
  tools, mapped to local CUDA 0/1 and 2 respectively. Model budgets enumerate
  only the weight devices. Tool placement and cap are exported as
  `CVPR_TOOL_DEVICE` / `CVPR_TOOL_GPU_BUDGET_MIB`; the job's `--tool-device auto`
  resolves that mapping. Read the launcher and per-run runtime records for actual
  assignments and budgets. Tools use float32 eval mode; these GPU-tool outputs
  have separate directories from the earlier CPU-tool FaithAct result.
- NF4 V13 and V14 are both complete at the same 1003520-pixel budget: each has
  online 8/16 and CF 7/16. V14's compact first proposal did not raise final
  accuracy. The existing `runtime/queues/v13_cf` worker is running job 003,
  `7b_dev_v15_atomic_repair16.json`, which runs online and CF sequentially.
  V15 keeps `compact_v1`, mandatory audits, and at most three repairs; its
  `atomic_v1` repair setting limits observation length to 180 and claim length
  to 140 characters. V15 online has completed at 9/16 (V14 8/16; CoT 10/16),
  recovering IDs 9/12 versus V14 and losing ID 7. Within V15, 14 valid initial
  answers go from 9 to 8 correct: no corrections, one regression on ID 7.
  Initial-answer-missing IDs 6/9 are separate. It has 7 uncertain finals,
  zero errors, and mean latency 95.448 seconds. V15 CF is still running without
  a completed result. Length limits do not guarantee semantic atomicity.
- Job 004, `7b_dev_v16_point_identity16.json`, follows V15 with the same model,
  prompt, repair, and input budgets. Its controller fix prevents the same
  normalized evidence question of an already closed point from reopening under
  another claim or kind. Failed query aliases stay closed but are never promoted
  to verified evidence: if query A fails and repair query B passes, close A but
  commit only B's supported claim as verified. This is a query-identity fix, not
  semantic paraphrase matching; no V16 result is available yet.
- V10 full-16 and V12 online were interrupted and preserved; V12 CF did not
  start. Their result markers say superseded. Do not relaunch these requests.

The first 128 cases are prompt development; the separate CoT validation slice
has completed, and matched BF16 online/CF validation is queued. Keep the selected
prompt and parser fixed for the remaining comparisons. The integration has
125 passing tests, including the V16 closed-point checks. HF milestone
`15ca6ded503bcecf3b699eb3b73bc58d52c8296d` published 261 files
through the earlier V13-online/brief-v4 stage; later outputs await the next batch.
That HF cutoff is the earlier parser-v1 stage and excludes the v2/v3 parser
reports and subsequent outputs. Source code
and reports are committed at research milestones; see Git history for publication state.

Historical sequence (not the active launch list):

- BF16: `7b_bf16_replication_v3.json`, full 765-case CoT/FaithAct reconstruction.
  Versions v1/v2 failed placement and are historical, not fallback launch choices.
- NF4: completed `7b_dev_002_pilot4_v3.json`, then v4 controller check,
  `7b_dev_003a_repair_regression_v5.json`, then
  `7b_dev_004_pilot16_v5.json` (declared standardized FaithAct answer format).
- 3B and the old full NF4 baseline are superseded diagnostics.

Submit to the appropriate resident worker inbox using a unique filename and
atomic rename from a non-JSON temporary filename. Never overwrite an active
request, reuse an output directory with changed code/config, or assume queue
process exit zero means the experiment succeeded: inspect worker results and
per-case outputs. Do not create another model instance if the matching resident
worker can run the request.
