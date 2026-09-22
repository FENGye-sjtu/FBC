# FaithAct score reproduction: explicit protocol, not a parity claim

Target: Table 2, RealWorldQA accuracy with Qwen2.5-VL-7B: CoT 70.1%,
FaithAct 74.5%. These are task accuracies, NOT Table 1 faithfulness scores.

The full 765-case BF16 reconstruction has completed: FaithAct 528/765 (69.02%)
versus its paired baseline 526/765 (68.76%) with the common reference-blind
extractor. It is 5.48 percentage points below the paper target. This is a
completed reconstruction with documented differences, not exact protocol parity.

Sources: user's FaithAct PDF; official upstream
https://github.com/lijunxian111/Faithful-First-RPA at
f0681f4c8d05fa415ee29f3906faa2bdcdf790ab.

## What is actually matched

- Official frozen Qwen2.5-VL-7B-Instruct checkpoint; unquantized BF16 run separate
  from NF4 engineering runs. Multi-GPU placement changes location, not weight dtype.
- All 765 pinned xai-org/RealworldQA examples, original images and answers.
- Released noun-extraction and evidence-conditioned prompts, blank-line CoT split,
  trained Poll checkpoint, 0.7 Poll + 0.3 Ground fusion, 0.4/0.6 bands and
  GroundingDINO thresholds 0.25/0.20.
- Maximum 512 generated tokens (consistent with reported Appendix H maximum).
- Both CoT and FaithAct receive the same images, precision and decoding settings;
  FaithAct reuses the exact saved CoT from its paired baseline.

## Unreleased or unresolved settings (do not invent)

- Released qwen_faith_inference.py imports qwen_inference.py, which is absent.
  The initial CoT JSONL and initial prompting adapter are also absent.
- The released runnable main block is a POPE path, not the RealWorldQA evaluator.
  RealWorldQA scoring/judge, random seed and exact image resolution are not specified
  in the released implementation inspected here.
- Our declared reconstruction uses greedy decoding, seed 0, 256–1280 visual
  tokens per image (200704–1003520 pixels), SDPA, Transformers 4.51.3, a
  documented CoT prompt, and reference-blind normalized answer extraction. The
  current formatting-permissive parser is reported alongside pinned official
  scorers; historical strict-parser results remain below.
  These are reconstruction settings, NOT proven original settings.
- The completed reasoned-v2 reconstruction used CLIP/DINO in eval-mode on CPU.
  The new layout-v5 queue uses a dedicated GPU for the same float32 tools,
  with separate output directories and recorded device/budget provenance.
  GPU execution is a declared hardware variant, not a replacement of the CPU result.
  The released poll function does not
  explicitly call eval(), which leaves dropout active. We use deterministic eval
  for the reconstruction and document the difference, not silently claim identity.
- Dataset packaging matters: the inspected OpenCompass RealWorldQA TSV
  (mirrored at mm-eval/VLMEvalKit) has reordered questions and MCQ choices for
  numeric questions. This is not interchangeable with the original mixed-format
  xai-org dataset. We retain the original xai-org version; the FaithAct release
  does not resolve which packaging/scorer generated Table 2. The OpenCompass
  checksum listed in VLMEvalKit is 4de008f55dc4fd008ca9e15321dc44b7. We therefore
  evaluate both packagings and never compare a score-only remapping with fresh inference.

## Score report required before claiming reproduction

Report completed n/765, accuracy, unanswered/error count, raw outputs, per-case
traces, full config/code hashes, and delta from 70.1%/74.5%. Diagnose a gap by
protocol component, without choosing settings on test labels until they hit the
published score. NF4/low-resolution results are always engineering diagnostics.

No external judge API is assumed. If a judge is later used, record its identity,
prompt, version and outputs; do not send ground-truth answers as extraction input.
We do not adopt random fallback answer assignment in some third-party evaluators.

## Completed prompt-packaging diagnostic and evaluator triangulation

Pinned evaluator sources: VLMEvalKit commit
`302cbce81a64b83b14fe3459a7390b3ff21b3e46` and AGS commit
`ab7452e08bb6827a245d8a53f2600960d219ac89`. The OpenCompass TSV MD5 is
`4de008f55dc4fd008ca9e15321dc44b7`; its 765 rows were mapped one-to-one
to the pinned xai images, then inference was rerun with the VLMEvalKit MCQ prompt.

| Inference packaging | Scorer | Correct / 765 | Accuracy |
|---|---|---:|---:|
| original xai mixed format | project final-answer parser | 479 | 62.61% |
| original xai mixed format | AGS `getResult_nlp.py` | 478 | 62.48% |
| VLMEvalKit MCQ, directive-dominated | stored/historical project parser | 539 | 70.46% |
| VLMEvalKit MCQ, directive-dominated | `committed_answer_v1` replay | 541 | 70.72% |
| VLMEvalKit MCQ, fresh inference (directive-dominated) | AGS `getResult_nlp.py` | 539 | 70.46% |
| VLMEvalKit MCQ, directive-dominated | VLMEvalKit `can_infer`, raw output | 513 | 67.06% |
| VLMEvalKit MCQ, directive-dominated | VLMEvalKit canonical final | 541 | 70.72% |
| VLMEvalKit MCQ, reasoned-v2 prompt | stored project parser v2 | 522 | 68.24% |
| VLMEvalKit MCQ, reasoned-v2 prompt | historical parser v3 replay | 524 | 68.50% |
| VLMEvalKit MCQ, reasoned-v2 prompt | `committed_answer_v1` replay | 526 | 68.76% |
| VLMEvalKit MCQ, reasoned-v2 prompt | AGS `getResult_nlp.py` | 523 | 68.37% |
| VLMEvalKit MCQ, reasoned-v2 prompt | VLMEvalKit `can_infer`, raw output | 456 | 59.61% |
| VLMEvalKit MCQ, reasoned-v2 prompt | VLMEvalKit canonical final | 526 | 68.76% |
| FaithAct, paired reasoned-v2 input | `committed_answer_v1` | 528 | 69.02% |
| FaithAct, paired reasoned-v2 input | AGS `getResult_nlp.py`, raw output | 461 | 60.26% |
| FaithAct, paired reasoned-v2 input | VLMEvalKit `can_infer`, raw output | 513 | 67.06% |
| FaithAct, paired reasoned-v2 input | VLMEvalKit canonical final | 528 | 69.02% |

The current server suite is complete with 765/765 rows and no missing IDs for
each file. Reports are [directive](reports/20260922_directive_eval.json),
[reasoned-v2](reports/20260922_reasoned_v2_eval.json), and
[FaithAct](reports/20260922_faithact_eval.json); server baseline snapshots and
disagreements are under `runs/eval_stage_20260922/{directive,reasoned_v2}`.
The 70.72% and 68.76% values replay fixed raw outputs; they are extraction
recoveries, not model or prompt improvements. Canonical-final scoring passes
only the reference-blind extracted answer to the pinned VLMEvalKit matcher, so
its agreement is not an independent fourth model result. Every method must use
the same frozen extraction protocol. Official AGS raw-output scoring remains
unmodified and separately labeled.

The subsequent `committed_answer_v2` adds a general affirmative inline-label
rule (for example, "the speed is B. 35") and unifies source selection across
the runner and all report tools: non-null raw text wins, including an explicit
empty response. An explicit inconsistent label is not changed using option
wording. All six completed BF16 files were replayed with pinned AGS/VLMEvalKit
under `runs/eval_stage_20260922_parser_v2`; every score in the preceding full-run
table and the prompt table below is unchanged. The published v1 snapshot and
its reports remain intact; new reports have `parser_v2` in their filenames.

The final frozen protocol is `committed_answer_v3`, SHA256
`49c5a365e342cf423be14fb4ef9d59e49e1bcf4f98f1036f101f0a71c1140a83`.
It preserves every valid v2 label and adds a terminal full-option recovery only
for previously unresolved weak conclusions; strong final fields still win.
The 20-file audit covers 3,794 rows (overlapping snapshots, not independent
examples). Earlier completed scores remain unchanged. V2's existing
false-positive limitations are inherited; this is a format protocol, not a
perfect semantic judge. Eight completed BF16 files have fresh official-scorer
reports in `runs/eval_stage_20260922_parser_v3` and `reports/20260922_parser_v3_*`.

The 70.46% result is numerically +0.36 percentage points from the paper's 70.1%
CoT number, with all 765 cases generated and no runtime errors. It is not yet a
CoT parity claim: inspection showed that VLMEvalKit's appended "Please select"
directive dominated the later reasoning instruction, so most outputs were only
an option letter and could not drive FaithAct's blank-line step extractor. The
completed reasoned-v2 run removed that embedded directive but appended another
"Return only the option letter" instruction. It still produced 602/765 outputs
with at most two whitespace-delimited words. Therefore neither run establishes
strict CoT parity, and their difference does not demonstrate that reasoning
causes an accuracy drop. AGS and the historical project parser agreed exactly
on the directive diagnostic. Pinned VLMEvalKit raw exact matching
marks 26 otherwise-correct outputs as `Z`: every disagreement is a false
negative caused by outputs that place the same option letter both on the first
line and after `FINAL ANSWER`; `can_infer_option` tests the first occurrence's
position. We report this stricter number rather than hiding it, but do not use
it to overwrite valid explicit final answers.

Mapping old xai-prompt outputs onto the VLMEvalKit labels without rerunning
inference gives only 278/765 (36.34%). This is a protocol-mismatch diagnostic,
not a model score: old outputs answer open questions with text/numbers while the
converted benchmark expects option letters.

Raw runs and evaluator records are under server paths
`runs/realworldqa_vlmevalkit_7b_bf16_cot_v1` and `runs/eval_protocols`.
An invalid FaithAct pass that reused these mostly direct outputs was stopped and
preserved after 269 cases. The replacement reasoned-v2 prompt run completed with 141
of 765 outputs containing at least one blank-line interior step that FaithAct can
inspect (mean 1.49 blank-line parts, maximum 7). The `committed_answer_v1` replay
recovers IDs 358 and 510 relative to parser v3; directive replay recovers IDs 122
and 577, with no correct-to-incorrect losses in either file. Earlier parser
counts remain in the table for provenance. The released-template FaithAct job
completed on the exact saved reasoned-v2 file. All 765 `original_cot` fields
match that cache; raw FaithAct SHA256 is
`57ec855ca4c9a175a5a26caaaf336a6e5d058af758c421b7c9d1d4c27b4bce36`.

| Completed paired subset | n | CoT correct | FaithAct correct | Corrections / regressions |
|---|---:|---:|---:|---:|
| all cases | 765 | 526 | 528 | 33 / 31 |
| cases with tool records | 141 | 84 | 85 | 17 / 16 |
| cases without tool records | 624 | 442 | 443 | 16 / 15 |
| cases with boxes actually included in the final prompt | 114 | 66 | 66 | 14 / 14 |

The overall net gain is only two answers (McNemar p=0.901); no reliable gain
can be claimed. Tool-record presence overstates delivered evidence: 336 of 575
noun records fell below the 0.4 fusion threshold and contributed only a noun
prefix, without box text. All 212 noun extraction calls completed without a
recorded format error. Nine originally correct answers became unresolved or
abstaining outputs. These observations locate bottlenecks in step coverage,
evidence delivery and final reasoning, but are not causal tool ablations.
Full subgroup IDs are in the [paired report](reports/20260922_faithact_paired.json)
and [diagnostics](reports/20260922_faithact_diagnostics.json). The large difference
between official AGS raw scoring and the common extractor remains separately
reported; do not change the extraction rule based on reference labels.

The three 128-case BF16 prompt development runs are complete:

| Prompt | Correct / 128 | Accuracy | Cases with inspectable interior evidence |
|---|---:|---:|---:|
| evidence_brief_v4 | 89 | 69.53% | 34 |
| evidence_adaptive_v4 | 80 | 62.50% | 69 |
| evidence_layout_v5 | 90 | 70.31% | 128 |

The same subset has 92 correct direct answers and 87 reasoned-v2 answers.
V5's literal blank-line layout produced no copied placeholders and retained
evidence in every case. It is selected as the evidence-producing candidate,
not the highest-accuracy direct-answer prompt. These are development results,
not 765-case scores or a recovered missing original prompt.

V5's separately selected validation IDs 128–255 are complete at 96/128 (75.0%),
also 96 under official AGS, with 127 inspectable outputs. On those exact IDs,
the directive baseline is 91/128 and reasoned-v2 is 89/128. The gains are +5
(11 corrections/6 regressions, p=0.332) and +7 (11/4, p=0.118), respectively.
This is encouraging validation evidence, not a significant or full-benchmark gain.

Cached V5 FaithAct on the development 128 is also complete: 88/128 (68.75%)
versus its exact initial CoT's 90/128 (10 corrections/12 regressions, p=0.832).
All 128 cases have tool records; 85 receive actual box text, scoring 57→52.
There were 129 noun calls, 309 noun inspections and no extraction errors.
Official AGS raw output is 23/128 (17.97%), raw VLMEvalKit 68/128 (53.125%),
and canonical-final VLMEvalKit 88/128. Preserve those raw-protocol results;
do not present the normalized result as an unmodified official score.
Some correct final labels contradict their own reasoning, so label recovery is
not proof of reasoning faithfulness. The fresh full-765 V5 CoT/FaithAct queue is
now running; its score remains pending.

The current integration passes 125 focused tests. V12 was stopped after an empty
visual-question field bypassed the exhausted-point key. V13 fills the missing
field from the public question stem. V13 and compact-initial V14 both finished
at 8/16 online and 7/16 with counterfactual verification, below matched CoT
10/16. V15's concise independent repair observations/claims completed online at
9/16, versus V14's 8/16 (wins 9/12, loss 7), still below CoT 10/16. Among its
14 valid initial answers, 9 were initially correct and 8 finally correct;
missing initial answers on 6/9 are reported separately. Its CF arm is running.
V16 is queued separately with
the same atomic-repair settings and a corrected closed-point budget: questions
already handled in a closed repair cycle cannot reopen as another step merely
by changing the claim or kind. This is normalized-string identity, not a
semantic paraphrase guarantee. After full FaithAct reproduction, the resident
BF16 worker will evaluate frozen V16 online/CF on the same 128–255 validation
slice in a separate directory; no result is claimed for that queued job.
NF4 engineering remains separate from BF16 reproduction. V7 had a 200704-pixel cap, versus 1003520 for
V9/V10 and the newer controls, so historical cross-version score changes do not
isolate method effects. The comparison harness now records these mismatches.
V9 CF completed at 8/16; V10's first-four smoke was 1/4 and its separate case 9
was wrong. The full-16 V10 job 006 was stopped with partial outputs preserved.
For neutral rereading, tentative fallback and the completed FaithAct analysis,
see the [stage record](reports/20260922_eval_repair_stage.md). A completed earlier
stage snapshot was published to HF at commit
`15ca6ded503bcecf3b699eb3b73bc58d52c8296d` (261 files, 8,491,867 bytes);
the [receipt](reports/20260922_publication.json) records its contents.
Later-finishing adaptive/layout, V13-CF and V14 results are intentionally
outside that immutable snapshot and remain available for the next batch.

## Separately labeled controlled-format variant

The NF4 16-case development job uses `--standardize-faithact-output`: the final
image-conditioned response gets the same explicit answer-format instruction as
CoT and our structured answer field. Noun extraction and all FaithAct tools are
unchanged. This addresses observed 1/0 vs Yes/No formatting failures without
reference-based relabeling. It is a declared variant, NOT the unchanged released
template or the BF16 reproduction run. The latter keeps the released template.
