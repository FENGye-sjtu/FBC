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
effects were unmeasured at that validation checkpoint because SSH was unreachable.

SSH subsequently recovered. The frozen-adapter v1 reader run completed all 768
updates in 489.317 seconds with peak allocated memory 3,085,314,048 bytes and
mean combined loss 3.244506. This combined loss is not comparable to the previous
positive-only mean loss. Its same-three-dev-case evaluation is running; see
`ACTIVE_RUN.md` for results. An earlier GPU-0 attempt OOMed during a competing
allocation; the successful training used GPU 1 and preserved all other jobs.

The next shared-model SFT also exports `efa_bindings` for native object episodes.
This teaches the decoder the same neutral identity interface before compact
reader alignment. It is restricted to native object sources and excludes whole
scene counting labels, held-out images and teacher-rejected trajectories.

The v1 three-case evaluation is now complete. Native reads remain 8/10 and
compact reads 6/10; EFA stays at 1/3. Cat is now correct and unverified, while
elephant is harmfully changed to horse after deletion/specificity failure.
The car question returns man with effective verified=true after task-query
drift. These outcomes establish neither reliable verification nor repair gains.
The complete result bundle is `efa/gqa_reader_objectives_eval_v1`, HF commit
`c88b458f931007b4ad82b3a4d23de30977b0ef72`. See `ACTIVE_RUN.md` for the next
mixed SFT job and its actual status handles.
