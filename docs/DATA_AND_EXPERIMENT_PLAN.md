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
3. **Reading and binding discrimination.** Preserve original images. Construct
   correct-region/wrong-value and wrong-object/right-value pairs only when native
   annotations or independent review establish incompatibility. Multiple regions
   may be required for relations, counting, chart scale/units and ownership.
   Missing annotations do not justify negative labels. Maintain same-value objects
   and duplicated evidence instead of forcing positive N/D gaps.
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

The annotation client is implemented, but the supplied key's HTTPS base URL and
vision model name are still needed. A small pilot should establish usable-episode
yield and cost before scaling. The deployed student never calls this API.

## Training sequence

- Completed engineering path: DeepEyesV2's LLaMA-Factory cold-start framework,
  Qwen2.5-VL-3B, four-bit LoRA, region reading/localization smoke. A ten-step run
  tests gradients and checkpoint saving; it is not a quality result.
- Next: align the eight-token source reader with native region/identity targets;
  train it jointly with the shared VLM adapters and measured sequence objectives.
  Its integration and 40-step positive-only alignment smoke are complete; the full
  objectives, sufficient training data and validation remain to be done.
- Then: SFT on reviewed step sidecars and predicted/incorrect/incomplete source
  views, using the same serialization as inference.
- Then: mine actual student errors with the bounded controller and SFT localized
  replacements. Train the implemented parent-revisit and suffix-regeneration path.
- Freeze weights and policy candidates before separate development/calibration
  evaluation. The current engineering thresholds have no calibration claim.

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
