# Native counting and premise deductions

`efa.clevr_programs` extends training beyond the GQA category chains. It selects
original CLEVR training questions with two filtered count branches and an integer
comparison, executes both counts against the complete original scene, and rejects
any disagreement with the native final answer. No teacher API or image editing is
used to construct these labels.

The resulting three-step interface is:

1. A visual count for the first neutrally described object set.
2. A visual count for the second set.
3. A premise-based comparison with explicit parents 1 and 2.

Each count uses the complete original scene as its source. Exhaustive counts,
including zero, require coverage of all potentially matching objects; point
coordinates are not misrepresented as annotated object boxes. The third step
has no image region of its own and inherits the two original-image sources.
These data do not demonstrate fine region localization or validate the current
count-family binding gate; both still require empirical study.

Sources are pinned to `dali-does/clevr-math` revision
`6a30110f887edd7edbad033275aa853ddd8c4a26`, `data/CLEVR_v1.0.zip`. The archive
contains original CLEVR PNG images, scenes, questions and license text. A bounded
HTTP-range reader retrieves its directory and selected members without downloading
the entire 19,021,600,724-byte archive. Source member CRCs, selected metadata
hashes, image hashes and revision are retained. Temporary signed download URLs
stay in memory only.

The first sample selects 256 unique image groups by a seeded image hash from
15,942 supported programs. The comparisons are 125 greater-than, 121 less-than
and 10 equality cases; native answers are 143 yes / 113 no. There are only three
template families, so this is a focused numerical extension, not broad deductive
coverage. Exact-image splits precede training exports; near-duplicate, benchmark
and template-family generalization checks remain pending.

The exporter adds `efa_premises`: text-only supervision with the exact runtime
premise-read prompt and the exposed parent claims. It includes neither images nor
invented verified statuses. `efa.mix_sft` combines GQA and CLEVR exports, checks
for train/held-out image collisions across sources and retains component counts.
`scripts/train_mixed_programs_2b.sh` prepares a new 2B SFT experiment with step,
region, visual-read and premise-read examples. It requires the existing baseline
evaluation to finish first; it has not been launched while SSH is unavailable.

Local validation currently passes 45 CPU tests, including native execution,
zero-count labels, reference disagreement, noncausal programs, premise-only
export, and correct bounded ZIP reads. Real-model learning and gate performance
are not established by these construction tests.

The complete `clevr_count_program_v1` artifact contains 256 original images,
768 steps (512 visual counts and 256 premise deductions), with 213 train /
27 dev / 16 calibration episodes. All selected programs agree with their native
answers. This build transferred 409,808,809 archive bytes before the final license
read. No complete 19 GB archive was stored locally.
It is published to private `QC1/FBC/efa/clevr_count_program_v1`, HF commit
`3d694fb95e7a7f52de40e2e0c3a59a19ac04fa40`, including the original license.

The actual combined SFT export with approved GQA train episodes has 1,023 step
examples, 810 source reads, 810 region targets and 213 image-free premise reads,
2,856 examples in total. Cross-source train/held-out image collision checks pass.
These are measured data/export counts, not training results.
