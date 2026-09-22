# FaithAct prompt and protocol audit

This audit answers why the released project has not reproduced the paper's
Qwen2.5-VL-7B RealWorldQA 70.1/74.5 task-accuracy pair. It distinguishes facts
in the paper, facts in the released source, and our declared reconstructions.

## What is actually released

- `template.py` includes `TEMPLATE_GCOT`: Grounding, then Reasoning, then a
  concise Answer. Thus a relevant reasoning prompt is released, but its name
  and the paper's separate Grounded-CoT baseline mean it cannot be assumed to
  be FaithAct's initial raw-CoT prompt.
- The released `qwen_faith_inference.py` never references `TEMPLATE_GCOT`.
  It imports `qwen_inference.py`, which is absent from the repository, and reads
  a precomputed CoT JSONL. Its runnable main is configured for POPE rather than
  RealWorldQA. Therefore the exact Qwen initial-generation adapter, question
  packaging and RealWorldQA task evaluator remain unreleased.
- FaithAct splits the saved response on blank lines and discards the first and
  last blocks. `TEMPLATE_GCOT` does not require that exact blank-line layout, so
  step coverage depends on unreported output behavior or missing adapter code.
- The release provides noun extraction and a whole-chain evidence rewrite. It
  does not implement the paper prose's immediate per-token/per-step generation
  boundary.

## Paper/code mismatches that affect reproduction

1. Appendix-E Algorithm 1 first creates the complete `R_raw`, then extracts all
   claimed objects, verifies them, and finally produces a complete `R_new`.
   This is batch rethink/refinement despite the main text's
   *verify-as-you-generate* wording.
2. Appendix-D's refinement prompt includes the original question, the original
   model response and tool information. Released `TEMPLATE_EOT` receives the
   question and tool information but omits the original response.
3. Section 5.1 reports GroundingDINO box/text thresholds 0.35/0.25. The released
   Qwen script uses 0.25/0.20. Both keep Poll/Ground fusion alpha 0.7 and
   absence/uncertainty/presence bands 0.4/0.6.
4. For a fused score below 0.4, the released Qwen code still appends a bare noun
   prefix to the final evidence block. Its final prompt simultaneously says all
   listed objects surely occur. This can turn an Abstain result into positive
   linguistic priming.
5. The released task-evaluation command is `eval_f_steps.py`, which measures
   perceptual-faithfulness scores, not the Table-2 RealWorldQA answer accuracy.
   No released script pins the Table-2 answer parser/judge.

## Controlled reproduction matrix now implemented

`faithact_gcot_official` preserves released `TEMPLATE_GCOT` as one candidate
prompt, stripping only the dataset wrapper's redundant leading `Question:` before
placing the payload into the template's own `Question:` field.
`faithact_paper_cot_literal` separately applies the paper Sec. 3.1 instruction
`Think step by step.` without adding a custom layout. Neither candidate is
called the recovered official FaithAct adapter.
`--faithact-prompt released_code` keeps public `TEMPLATE_EOT` and the
Qwen script thresholds 0.25/0.20. `--faithact-prompt paper_appendix` includes the
original response, explicitly labels low-confidence objects absent and is queued
with paper thresholds 0.35/0.25. Manifests record every axis. Neither variant is
called exact paper parity because the initial adapter and Table-2 evaluator are
still missing.

The released-GCOT candidate has completed all 128 pairs. Its initial chains are
78/128 (60.94%) with 128/128 inspectable interiors; after released-code FaithAct
they remain 78/128 (60.94%), with 17 wrong-to-right corrections and 17
right-to-wrong regressions (McNemar p=1.0). Thus this prompt/rewrite pair gives
zero net gain, not the paper's +4.4 points. The same slice's direct baseline is
92/128, so the template is runnable and step-compatible but is strong evidence
against treating `TEMPLATE_GCOT` as the missing Table-2 CoT protocol.

The evaluator split also locates formatting failures. For the initial GCoT,
committed/canonical-final/AGS scores are 78/78/77, while raw VLMEvalKit is 49.
For the FaithAct rewrite, committed and canonical-final both score 78, raw
VLMEvalKit scores 58, and raw AGS scores only 16. AGS's released regex captures
the text after `final answer` before stripping a following `is`; for the released
rewrite's common `Thus, the final answer is A. ...` form this leaves a phrase
instead of an option label. GCoT's `Step 3: Answer: A. ...` happens to fit that
rule. VLMEvalKit similarly loses long-form outputs but returns to 78 once it sees
only the same reference-blind canonical final answer. These raw columns diagnose
format compatibility; they must not replace the common answer-accuracy column.

The 17 regressions include five explicit abstention/no-option outputs. Another
clear failure treats detector pixel coordinates as metric distances: the rewrite
subtracts x coordinates, calls the result meters, then guesses the nearest
offered option. This is a protocol-level reason not to paste raw box coordinates
into our next-step reasoning context. Our controller uses boxes only to construct
views and keeps coordinates out of the verified evidence ledger.

The paper-appendix reconstruction on the same GCoT cache is also complete at
78/128. Against the original GCoT it has one correction (ID 29) and one
regression (ID 96), p=1.0. Against the released rewrite it changes 44 extracted
answers: it recovers all 17 released-rewrite regressions plus one other error,
but loses all 17 released-rewrite corrections plus one other correct answer.
Including the original response and prohibiting coordinate/confidence repetition
therefore acts mainly as an answer-preserving stabilizer, not an accuracy gain.
It does improve parseability: valid labels rise from 116/128 to 125/128 and
unresolved MCQs fall from 12 to 3. Raw AGS/raw VL/canonical scores are 20/56/78.
This separates format stability from semantic improvement.

The paper-literal prompt pair is complete. Initial `Think step by step.` scores
94/128 (73.44%) with 126 valid labels and all
128 outputs supplying inspectable interior blocks. On the identical IDs this
exceeds direct 92/128, layout-v5 90/128 and released GCoT 78/128. Raw AGS and
raw VLMEvalKit score only 35 and 58 because their raw-format rules see the full
chain; canonical-final VLMEvalKit agrees with the committed score at 94. This
is the strongest initial-chain candidate so far and supports the paper's terse
Sec. 3.1 description over the released Grounded-CoT template. It is still a
128-example development result, not the paper's full official reproduction.
The paired released-code FaithAct rewrite falls to 87/128 (67.97%): 12
corrections, 19 regressions, net -7 (McNemar p=0.2810). It has 122 valid labels,
128 tool cases, 738 noun inspections and 112 cases with box text. Raw AGS/raw
VL/canonical scores are 24/69/87. Thus the best initial prompt does not recover
the paper's 70.1→74.5 improvement; the released whole-chain evidence rewrite
instead damages this stronger starting point in the aligned development run.

The completed `evidence_layout_v5` run is a project-designed input layout, not the
official FaithAct prompt. Its paired baseline is 542/765 (70.85%) and the released-
template FaithAct port is 509/765 (66.54%): 57 corrections, 90 regressions, net
-33 (McNemar p=0.0081). In the 546 cases that received actual box text, accuracy
changed 379 to 349 (40/70, p=0.00545); in 218 tool cases without box text it
changed 162 to 159 (17/20, p=0.743). This association is not a randomized causal
box ablation. Concrete failures include a count response whose prose says two
bags but whose final label maps to three, and an orientation rewrite that
introduces an unsupported front/back conclusion. These motivate our separate
evidence-then-deduction controller and do not justify tuning on reference labels.
Parser v4 adds a reference-blind formatting recovery for ID 728's Markdown-bold
final marker, making the common and canonical-final scores agree at 510/765
(66.67%). The baseline remains 542/765, so the paired result is 57 corrections,
89 regressions, net -32 (McNemar p=0.0101). This one-answer parser correction is
reported separately and is not a model improvement. Raw AGS (125) and raw
VLMEvalKit (413) remain much lower because they score the full explanatory string
under their own format rules.

## Next measurements

1. Preserve the completed full V5 job and its frozen code snapshot.
2. Preserve all three completed 128-case protocol pairs and use the paper-literal
   94/128 chain as the strongest open reconstruction baseline for our controller.
   with the common committed parser, raw AGS and raw/canonical VLMEvalKit outputs.
3. Inspect step coverage, low-confidence noun delivery, box/no-box paired changes,
   answer-mapping contradictions and exact output hashes before deciding which
   FaithAct row is the primary reproduction baseline.
