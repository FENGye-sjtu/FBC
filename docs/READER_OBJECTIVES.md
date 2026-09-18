# Native reader discrimination

The manuscript's reading and rejection objectives are now wired into
`efa.train_reader_objectives`. They use the shared VLM and source resampler;
there is no additional factuality classifier or deployed teacher.

`efa.discrimination` builds the data only from approved training episodes. Each
record retains original crops, native object IDs, the neutral query, a positive
value, identity text and any known incompatible category strings. It never treats
missing scene-graph annotations or failed teacher review as negative evidence.
The first policy contains disjoint animal species and vehicle kinds; it excludes
overlapping human roles, material mixtures, subclasses such as car/taxi and
potentially coexisting relations. These are explicit native-category assumptions,
not independent visual adjudication of every negative or controlled image edits.

Published data `QC1/FBC/efa/gqa_reader_objectives_v1`, commit
`6114f79c0c3b3b521c63cf1140b11ad26325e2f9`, contains 384 read/identity records
from 126 train images and 126 episodes. Only 27 atoms qualify for this negative
policy, yielding 54 wrong strings. Class frequencies are reported in its manifest;
they are not balanced. This is an initial training subset, not adequate evidence
of broad discrimination. None of the dev/calibration images enter this artifact.

The trainer optimizes summed correct-answer sequence NLL, summed identity NLL
with weight 0.2, and mean whole-string unlikelihood over eligible wrong strings
with weight 1.0. The latter is `-log(1-exp(logp_wrong))`, including EOS, not a
length-normalized pseudo-probability. Loss weights are experimental settings.
Each term backpropagates separately before one optimizer update to avoid holding
all source-view activation graphs at once. Identity prompts exactly match the
runtime binding prompt. No target-deletion or distractor margin is forced.

By default only the reader is updated. `--train-adapter` also updates existing
shared LoRA weights, with a separate learning rate and a saved matching adapter.
Inference rejects a reader paired with a different adapter hash. Explicit training
initialization may reuse a reader for realignment; visual precision still must
match. This joint mode has not yet been tested with the real GPU model.

`scripts/train_reader_objectives_2b.sh` prepares the next frozen-adapter experiment
after the existing v2 baseline evaluation completes. It initializes from its
positive-aligned reader, adds the identity/negative objectives for 768 updates,
then evaluates the same three dev images. The source-view objectives remain part
of Stage 1: predicted/wrong/incomplete source training, natural student rollouts,
revisit supervision and policy calibration are still required.

Validation: 39 tests pass in a local CPU PyTorch 2.6 environment, including
unequal-length sequence loss values, gradient directions, unchanged neutral
prefixes, held-out exclusion and invalid negative provenance. The CLI imports
and launch shell syntax also pass. GPU execution, memory and actual accuracy
effects are unmeasured because the server SSH port is currently unreachable.
