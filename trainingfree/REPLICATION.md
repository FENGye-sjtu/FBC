# FaithAct score reproduction: explicit protocol, not a parity claim

Target: Table 2, RealWorldQA accuracy with Qwen2.5-VL-7B: CoT 70.1%,
FaithAct 74.5%. These are task accuracies, NOT Table 1 faithfulness scores.

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
  documented CoT prompt, and strict reference-blind normalized answer extraction.
  These are reconstruction settings, NOT proven original settings.
- Our CLIP/DINO inference is eval-mode on CPU; the released poll function does not
  explicitly call eval(), which leaves dropout active. We use deterministic eval
  for the reconstruction and document the difference, not silently claim identity.
- Dataset packaging also matters: the inspected OpenCompass RealWorldQA TSV
  (mirrored at mm-eval/VLMEvalKit) has reordered questions and MCQ choices for
  numeric questions. This is not interchangeable with the original mixed-format
  xai-org dataset. We retain the original xai-org version; the FaithAct release
  does not resolve which packaging/scorer generated Table 2. The OpenCompass
  checksum listed in VLMEvalKit is 4de008f55dc4fd008ca9e15321dc44b7.

## Score report required before claiming reproduction

Report completed n/765, accuracy, unanswered/error count, raw outputs, per-case
traces, full config/code hashes, and delta from 70.1%/74.5%. Diagnose a gap by
protocol component, without choosing settings on test labels until they hit the
published score. NF4/low-resolution results are always engineering diagnostics.

No external judge API is assumed. If a judge is later used, record its identity,
prompt, version and outputs; do not send ground-truth answers as extraction input.
We do not adopt random fallback answer assignment in some third-party evaluators.

## Separately labeled controlled-format variant

The NF4 16-case development job uses `--standardize-faithact-output`: the final
image-conditioned response gets the same explicit answer-format instruction as
CoT and our structured answer field. Noun extraction and all FaithAct tools are
unchanged. This addresses observed 1/0 vs Yes/No formatting failures without
reference-based relabeling. It is a declared variant, NOT the unchanged released
template or the BF16 reproduction run. The latter keeps the released template.
