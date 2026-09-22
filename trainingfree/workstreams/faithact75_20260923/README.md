# FaithAct reproduction and Evidence-First 75+ workstream

This directory isolates the 2026-09-23 continuation from all earlier runs.
Existing outputs are read-only inputs. New server snapshots, queues, logs and
results must live under:

`/data_cinema/zqc/CVPR/workstreams/faithact75_20260923`

## Non-negotiable protocol

- Qwen2.5-VL-7B remains frozen.
- RealWorldQA comparisons use the same IDs, image budget, precision and
  reference-blind `committed_answer_v4` parser.
- FaithAct reproduction rows distinguish released code, paper-stated settings,
  and declared reconstructions. Missing upstream artifacts are never silently
  filled in and called official.
- Evidence First generates one atomic evidence point, audits it, repairs it no
  more than three times, records unresolved points as uncertain, and continues.
- Existing run directories and queues are never modified or resumed with new
  code. Every candidate gets a new immutable snapshot and output directory.
- The first 128 RealWorldQA examples are prompt-development data. IDs 128-255
  are the first validation slice. No candidate is called 75%+ until a fixed
  protocol measures at least 96/128 on validation or 574/765 on the full set.

## Current evidence

- Paper-literal `Think step by step.`: 94/128 on development.
- Released-code FaithAct rewrite of those exact chains: 87/128, with 12
  corrections and 19 regressions.
- Full layout-v5 reconstruction: 542/765 before and 510/765 after FaithAct under
  parser v4. This is not the missing official Table-2 protocol.
- V20 Evidence First: 10/16 on the NF4 engineering slice, identical to its
  paper-literal CoT anchor. V21 is still running and must not be overwritten.

## Decision sequence

1. Inventory all completed outputs and quantify paired complementarity without
   using references in any proposed selector.
2. Diagnose FaithAct corrections and regressions by evidence availability,
   parser state, confidence bands, and answer changes.
3. Freeze at most two reference-blind candidates on development.
4. Run them on IDs 128-255 in an independent queue and accept a 75%+ claim only
   from that fixed validation protocol.
5. Run the selected method on all 765 cases and report corrections, regressions,
   uncertainty, errors, latency and tool calls in addition to accuracy.

