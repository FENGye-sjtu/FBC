# Native GQA step supervision

`efa.gqa_programs` compiles explicit GQA programs and scene-graph annotations into
numbered visual claims. It uses the pinned Visual-CoT metadata already downloaded,
the original Visual Genome JPEGs, and the native GQA training scene graphs. A seeded
sample retains one program per original image. It does not ask a teacher to invent
intermediate facts or label runtime audits.

Supported chains select an object, filter an annotated attribute or geometric
position, traverse a directed native relation, and query an annotated category or
property. A selection becomes an existence observation; the final category is not
inserted into its neutral query. Relation direction must match an actual graph edge.
Program dependencies are preserved. Unsupported operations and unresolvable labels
are counted instead of guessed. Source IDs and the exact native evidence accompany
every atom, but are not included as privileged inputs in SFT conversations.

The first version contains 512 original images / 512 trajectories / 1,629 steps:
410 training trajectories, 55 dev, 47 calibration. All splits use decoded-image
hashes; formal benchmark and near-duplicate clearance remains pending. Programs
passing graph consistency are still source annotations, not independently verified
image claims. In particular, small objects and ambiguous relational references
remain difficult to read even with correct native boxes.

Source review now checks target-only evidence separately from target plus context.
Both available views must support the atom. A whole-image context can no longer
hide an unsupported target crop. The first 64 trajectories yielded 21 teacher
approvals: 20 train / one calibration, 64 approved steps, no API errors. Recorded
usage is 159,083 prompt and 28,944 completion tokens. These are teacher judgments,
not ground truth, fitted gates or student PASS outcomes.

Inspection of rejections identified missing anchors in composite referents: a
final query such as "object that person is holding" needs the person as well as the
object. Version 2 brings every referenced ancestor's native region into the target
source set and clarifies relation direction in the referring text. It recompiles
the same 512 original images and programs without changing answers or inventing
regions. All 512 cases have now been reviewed, reusing only identical cached API
requests from the 64-case pilot. There are 156 approved trajectories: 126 train,
11 dev and 19 calibration, containing 475 approved steps. No API errors remain.
Recorded cached-response usage is 1,429,066 prompt plus 241,219 completion tokens.
The complete frozen dataset is `efa/gqa_native_review_v2_complete`, HF commit
`e19bda2b491706c57cabe7873767c7258eb6311a`; it is also uploaded to the server.
The eligible train split contributes 384 atoms/steps. Its 126 final targets are
all category queries; the other steps are presence, native relations and a few
attributes. This dataset does not yet cover numerical reasoning, counting or
explicit premise-based deductions. Those require additional native program or
table supervision before claiming the complete training distribution.
On the same first 64 cases, approvals changed from 21 to 23; this is only a small
annotation-protocol comparison, not evidence of a student-model gain. The first
128 completed version-2 reviews are frozen separately: 38 approved trajectories,
33 train / one dev / four calibration, and 116 approved steps. The parent 512-case
was incomplete at snapshot time. HF version `efa/gqa_native_review_v2_first128` is commit
`86b94d1e02f09197be3d058e655f762be0371f1b`.
The first 256 completed reviews are also frozen: 77 approved, 66 train / three dev
/ eight calibration, 233 approved steps; HF `efa/gqa_native_review_v2_first256`,
commit `1076ae27287a2d129be5ffde356c4a79afd5c27a`.

The SFT exporter reads reviewed teacher trajectories for engineering runs. The
review folder stores native candidates in `source_native_episodes.jsonl`, which is
provenance only; it cannot silently reintroduce rejected candidates into training.
The compact-reader trainer now supports every visual atom and multiple source
regions from the selected episode annotation tier. It excludes held-out and
rejected episodes. Positive alignment is still not the full discrimination objective.

`scripts/train_gqa_2b.sh` launches a fresh Qwen2-VL-2B adapter from the base model on
the 20 approved training trajectories. It uses rank-16 q/k/v/o LoRA,
128 SFT updates, accumulation two, learning rate 2e-5, and a 50,176-pixel budget;
then 256 positive multi-region reader alignment updates. The ZwZ-trained adapter
and reader are not initialization sources for this run. Completion and quality
must be measured from its artifacts. This is an intermediate supervised pipeline,
not finished paper training or evidence of CVPR-level results. This run completed
128 optimizer updates: training loss 0.789799, runtime 105.404 s. Multi-region
alignment completed 256 updates on 61 visual atoms from the 20 training episodes
in 81.719 s, with peak allocated memory 2,442,359,296 bytes. Training fit is not
held-out performance; an episode-level evaluator now measures source reads and
the full generated/audited trajectory on the retained calibration image.

That evaluation completed on `gqa-native-6019` (reference: train). Both readers
got only the existence claim correct (1/3 atoms). Native answers were yes / in /
car; compact answers were yes / on / suv, against yes / in front of / train.
Direct answering returned Car. The full controller generated only one step,
performed three repair rounds / ten audits, and returned the incorrect unverified
fallback make. This does not demonstrate learned multistep behavior. The initial
audit had valid routing and failed support/deletion, rather than binding; the
underlying source reader and premature completion require further work.
The full trace, component outputs, training metrics and configuration are published
at `efa/gqa_heldout_2b_v0`, HF commit
`edb50cb48acad2f56af2de093adacc58382f13f9`.

Inspection of LLaMA-Factory's installed quantization code did not reveal a visual
exclusion in its BitsAndBytes configuration. A matched original-model probe on
the identified calibration image returned Car under all-module NF4 at both
50,176 and 200,704 pixels; BF16 returned Train/train at both resolutions.
The corrected mixed-precision probe also returned Train/train at both resolutions,
with actual vision modules `Linear`, language modules `Linear4bit`, and peak
allocated memory 2,527,849,472 / 2,575,278,592 bytes. This is a single-case
perception diagnosis, not evidence of an aggregate benchmark improvement.

The first mixed-precision attempt excluded only `visual` and failed in `lm_head`:
an explicit exclusion list replaces the Transformers defaults, and this model
ties its output head to floating embeddings. The corrected policy excludes both
`visual` and `lm_head`. `efa.sft_entry` applies this policy within the training
process without editing LLaMA-Factory, and verifies actual module types before
optimizer updates. Reader checkpoints record encoder precision; old NF4-vision
readers require explicit `--quantize-vision` replay and cannot silently load into
the new default. New SFT and reader training start from fresh weights.

`scripts/train_gqa_mixed_2b.sh` uses all 126 approved train trajectories, 1,200
SFT updates, fourfold sampling of numbered-step examples, then 768 multi-region
reader alignment updates. Its evaluator selects three dev episodes by hash.
It compares direct answering, the same checkpoint's numbered steps without audits,
and full EFA on the same images. The unaudited path exposes prior values as
unverified, supplies no reference regions, and records unsupported parent-revisit
requests as failures instead of invoking hidden audits. These comparisons are not
compute matched; traces retain individual forward-call records.
Training and evaluation outcomes must be read from the run artifacts; the
single-case loading correction does not establish step reasoning or audit quality.

The first SFT precision callback revealed another configuration error: the
installed LLaMA-Factory expects `quantization_method: bnb`; the earlier value
`bitsandbytes` silently skipped quantization. Run `gqa_sft_2b_v1` was stopped
before optimizer updates when no `Linear4bit` language modules were found.
Consequently earlier SFT runs using this configuration must not be described as
verified NF4 training. Their NF4 inference/reader alignment artifacts remain
historical measurements, with a train/inference precision mismatch. The corrected
configuration uses `bnb`, requires a generated quantization config, and records
actual module types. The fresh run is `gqa_sft_2b_v2`, followed by
`latent_gqa_2b_v2` and `gqa_heldout_2b_v2`; the failed v1 directories are preserved.

Published private QC1/FBC data versions:

- `efa/gqa_native_program_v1`: `e10218bdd8ad1906d41d510794a8b8057f90bbff`.
- `efa/gqa_native_program_v2`: `1847d840ed3a54007ba5cfbc009c50fbab722e80`.
- `efa/gqa_native_review_v1`: `3e975204627aa09974a896343d4adc45f6af57a8`.

All 35 tests passed on the server; the local suite skips two optional torch tests.
Precision diagnostics are published as `efa/vision_precision_2b_v0`
(HF `01e921e61ed4fac9e146b1e5aebf6c5f9d9b06bc`) and `efa/vision_precision_2b_v2`
(HF `c4419dd81903781efeea6ec212a552a9d3eefca7`).
Tests cover relation direction, missing source edges, inherited
source anchors, rejected/held-out exclusion and target-only review failure; they
do not establish model correctness or calibration.
