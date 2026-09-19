# Data and experiment execution plan

The target is the manuscript's admission of each observation into the next step:
audit the existing value first; search alternatives only after failure. Training
must teach this interface, not just produce long rationales with decorative boxes.

## Dataset layers

1. **Native region skills.** Use ZwZ original images, boxes and canonical answers
   for localization/reading. Its reference values retain teacher-consensus status.
   Use native GQA/Visual Genome annotations for identities, attributes and relations;
   use chart tables and checked programs for numeric/deductive targets. Initial
   ZwZ seed contains 18 records; it is an engineering artifact, not the target scale.
2. **Grounded step episodes.** Reuse suitable DeepEyesV2/DeepEyes questions and
   solution structures plus Visual-CoT/GQA programs. Parse a minimal sequence of
   useful steps. An independent value-blind request locates each visual atom using
   only neutral identity and predicate. Boxes attached to the whole question are
   never automatically copied to every intermediate step. Deduction atoms name
   explicit parents and inherit their original-image evidence.
3. **Automatic source audit.** Preserve original images. After value-blind routing,
   use the same frozen VLM to teacher-force the incumbent on native target/context,
   deletion and control crops. Calibrate PASS/REVIEW thresholds on a held-out
   calibration split; do not train a separate verifier or teacher-label its verdicts.
   Compact readers and reviewed wrong-value pairs remain optional ablations only.
4. **Student errors and repairs.** Run the student one step at a time, retain its
   real incumbent, real B/S/N/D scores and all controller actions. Only REVIEW
   permits up to three rounds of three proposals. Retain no-change repairs,
   harmful corrections and exhausted unverified continuations. Never fabricate
   these trajectories with teacher-written PASS labels.

Images and near-duplicate groups must be split before generated variants. Formal
evaluation images/answers must be excluded before teacher annotation. Current
exact-hash splits are only the first layer; benchmark/near-duplicate clearance is
still pending. Count region tasks, complete reviewed episodes, and real repair
episodes separately in each release.

## Offline API use

Stage A generates/parses structured steps from the original task. Stage B starts
a new conversation and receives only neutral visual queries and the original
image. Stage C will independently check claim/region support against native
labels, source crops and adjudication. Record model, source revision, request
hash and usage. Keep teacher labels separate from measured student scores.

The user confirmed DashScope's official compatible-mode endpoint, and successful
authenticated model discovery confirmed `qwen3.5-flash-2026-02-23`. The pilot uses
that pinned model with thinking disabled. Fresh-context crop checks retain
teacher-pseudo-label status; they are not ground truth. Qwen teacher coordinates
use a declared 0..1000 frame and are converted to original pixels. Native boxes
are supplied as answer-free candidates. The deployed student never calls this API.

## Primary training sequence

- Completed engineering path: DeepEyesV2's LLaMA-Factory cold-start framework,
  Qwen2.5-VL-3B, four-bit LoRA, region reading/localization smoke. A ten-step run
  tests gradients and checkpoint saving; it is not a quality result.
- SFT one complete alternating trajectory per reviewed episode. The trajectory
  keeps all earlier user/assistant turns; value-blind routing remains a linked,
  isolated call by the same VLM and deterministic controller.
- Evaluate the resulting adapter directly with native original-image crops and
  automatic teacher-forced evidence scoring. No resampler or negative-pair
  continuation is required for the primary result.
- Mine actual student errors with the bounded controller and SFT localized
  replacements. Train the implemented parent-revisit and suffix-regeneration path.
- Freeze weights and policy candidates before separate development/calibration
  evaluation. The current engineering thresholds have no calibration claim.

The completed eight-token reader and reviewed positive/negative continuation are
preserved as exploratory ablations. Their six-case run did not establish a gain and
must not be used as the primary checkpoint without new evidence.

## Evidence required for the paper

First establish that the gate detects the student's natural wrong premises and
that failure-triggered repair improves final answers over data-matched SFT. Use
the same backbone and examples for base CoT, SFT, post-hoc checks, online checks,
always-on proposals and EFA. Measure all actual forwards, crop encodings, latency,
memory and fallback outcomes; candidate counts alone are not compute matching.

The manuscript's initial benchmark protocol is MMStar, MMMU validation, MathVista
testmini, ChartQA test, TextVQA validation and HallusionBench. Add larger backbones
after the 3B path works. Report ordinary accuracy including every fallback,
verified coverage/error, unverified accuracy, harmful repairs and uncertainty
inheritance. Replace illustrative tables only with measured reproducible results.
