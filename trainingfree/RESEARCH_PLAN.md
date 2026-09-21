# Implementation priorities agreed with the user — 2026-09-21

## Fixed story

A speculative visual premise can contaminate subsequent reasoning. The proposed
inference controller therefore does not expose a candidate to the next reasoning
step as a verified premise until evidence verification accepts it. A rejected
candidate is repaired at the same boundary for at most three repair rounds after
the initial check. If still unresolved, mark that point uncertain and CONTINUE;
never silently treat it or its dependent conclusions as verified. This is not
full-chain generation followed by rethink (latest user clarification).

The backbone remains frozen. Start from FaithAct's released inference/tool code,
not the older project's SFT pipeline. Our lightweight verifier reuses the frozen
VLM, independently localizes image evidence, and measures targeted counterfactual
dependence relative to matched control occlusions. Future rollback is optional;
it must not delay the initial accuracy experiments.

"Accepted by the verifier" is measurable; "guaranteed true" is not. We will not
claim that all earlier work is post-hoc: online verification has related work.
The contribution must be the evidence gate, contamination control and measured
accuracy/cost improvements, rather than only changing when a prompt is called.

## Immediate execution order

1. Make real frozen-model generation, tool calls, structured steps and audits run.
2. Paired 16-case RealWorldQA engineering run: CoT, FaithAct port, online,
   online-counterfactual. Inspect actionable failures and correct implementation.
3. Run the complete 765-case CoT/FaithAct baseline. Compare with published 70.1%
   and 74.5% only after reconciling prompting, generation and answer evaluation.
4. Expand our method to the same frozen benchmark with fixed development settings;
   report full-set accuracy (including failed/unanswered cases), coverage, calls
   and latency. Add MMStar next, reusing existing public input images read-only.
5. Improve actual failure modes: grounding, visual contrast, stopping/answer
   synthesis and repair. Add rollback only after the online controller is useful.

Do not repeatedly add peripheral controls before obtaining real score tables.
Keep the necessary comparison small: post-hoc vs stepwise, then stepwise vs CF.
Do not tune thresholds against final test labels; separate engineering examples
and a declared development split from the final score report.

## Coordination between computers

Canonical code: `FENGye-sjtu/FBC`, `main`, directory `trainingfree/`.
Do not commit someone else's dirty `efa/` working tree. Before changes fetch main;
sync completed research milestones in batches, not every edit, without force.
Runtime artifacts belong
in HF dataset `QC1/FBC` under a separate `trainingfree/` prefix when published.

Server layout (all new writes confined here):

- `/data_cinema/zqc/CVPR/code/trainingfree`: current implementation.
- `/data_cinema/zqc/CVPR/upstream/FaithAct`: unmodified pinned source.
- `/data_cinema/zqc/CVPR/models`: weights or verified read-only weight references.
- `/data_cinema/zqc/CVPR/data`: new benchmark materialization.
- `/data_cinema/zqc/CVPR/runs`: immutable named experiments and per-case traces.
- `/data_cinema/zqc/CVPR/logs`, `runtime`, `cache`, `tmp`: isolated runtime state.
- `/data_wyn/zqc/CVPR`: reserved; avoid large artifacts because disk is nearly full.

Existing Python packages and public pretrained weights may be read from older
directories. Never install into their environments, rewrite their caches, kill
their tasks, or overwrite their files.

GPU policy: ignore utilization; share occupied GPUs whenever memory fits. Use
quantization/small engineering models for tight memory, clearly separate those
numbers from original-precision replication. Do not wait for an idle GPU.
