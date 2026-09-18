# Teacher and data pilots, 19 September 2026

The architecture is unchanged: stop at each step boundary, acquire evidence from
a neutral query, audit the incumbent, and propose corrections only after failure.
These experiments supply supervised interfaces; they are not measured runtime gates.

## What changed

The DeepEyesV2 source is ordered by task type. The first 64 selected visual rows
were all counting questions. A seeded shuffle now draws across the complete
part-0 index. The server extracted 128 records / 126 images. A locally constructed
32-record / 32-image slice used 12,569,989 HTTP range bytes from the 10.27 GB archive;
it is published at `efa/deepeyesv2_mixed_v1`, HF commit
`328bb92113c03319fb24070b45b761654098cfb0`.

Teacher JSON may omit a redundant `Step N:` text prefix or return a numeric value
as a JSON number. These are normalized without changing semantic values or the
cached original response. The explicit `deduction` mode alias is normalized to
`premise`, still requiring existing parents. Bare `relation` predicates are
rejected because the reader cannot recover their reference entity from main-trace
text. Relational queries must identify both participants before source acquisition.

Reviewed episodes now supervise three paths: next-step generation, neutral region
selection, and cropped source reading. Reader prompts and original-frame coordinates
match the actual backend, including multiple independently cropped input images.
Native property examples and compact-reader alignment use that same read interface.

## Measured teacher trials

Under the same pre-program-prompt protocol on eight counting tasks, Flash
`qwen3.5-flash-2026-02-23` produced two structured episodes and one teacher approval
(calibration split). `qwen3-vl-plus` produced eight episodes and five approvals,
four train and one calibration. Two transient review failures were retried using
cached annotations. All five were single-step; this comparison does not establish
multistep quality or dollar cost superiority. It predates deduction-alias handling.
Plus was selected provisionally for the next modality trials.

Cached usage: Flash 15,563 prompt + 2,947 completion tokens; Plus 32,841 prompt +
2,107 completion tokens. Reused cache tokens are not newly billed usage. Plus data:
`efa/deepeyesv2_plus_comparison_v2`, HF commit
`64cbd72e8d200eafe7a53a1c7c5647e1a55abd37`.

Visual-CoT/GQA source: 32 questions / 30 original Visual Genome images, each with
at least four native program operations. Programs and native region candidates
retain source provenance; the question-level box is never copied to every step.
The source revision is `223d2d8c1146fda2bb918801b8276c587b78b61c`, published at
`efa/viscot_gqa_program_v1`, HF commit
`3ee4b3c837c7b4b791dc59f60c779e35e9c4eacc`.

The revised relational prompt produced 16 structured GQA episodes; three passed
same-teacher source review, all multistep (two train, one calibration). The train
export contains five steps, six region targets, and six crop reads. Published at
`efa/viscot_gqa_plus_v2`, HF commit `1b5d0e35807aa125339a439a75864224d7c22d7c`.
Small/ambiguous crops and incorrect bindings remain major rejection reasons.
An earlier two-approval trial had a bare relation query and is retained only as
an earlier protocol artifact; its approvals do not imply current eligibility.

ChartQA arithmetic source: 16 questions / 16 images from the training split,
revision `b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5`. The first trial produced nine
structured episodes and one reviewed three-step training episode: read 88, read 9,
then subtract to get 79 using both explicit parents. Seven cases failed annotation
format/atom constraints. A cache-reusing trial with deduction-alias normalization
is tracked separately. Reference answers can induce incorrect supporting values;
therefore exact final-answer agreement alone never approves a trajectory.

The alias-normalized ChartQA trial produced 12 structured episodes and the same
one approved three-step training episode; four atom-count rejections remained.
Published at `efa/chartqa_plus_v2`, HF commit
`e7130e46ac68df5746b76bb388c2df7ffe06b7b8`. Combined with GQA, current approved
multistep training exports contain eight step examples, eight crop reads and
eight region targets. GQA's third episode remains held out.

## Training and remaining work

The next native-region run is `scripts/train_region_pilot.sh`: 200 SFT optimizer
steps, Qwen2.5-VL-3B, four-bit LoRA rank 8 on q/v projections, image budget 50,176,
sequence limit 512, accumulation 4; followed by 256 compact-reader alignment steps.
All data remain engineering-only pending benchmark/near-duplicate clearance.
The first launch exited before model loading because no GPU met the 7.5 GB free
threshold. The adjusted run requires 5.5 GB and uses the shorter sequence limit.
Actual completion/losses must be read from the server output, not inferred here.
The completed region export contains 718 crop reads, 250 region targets and 250
single-step format examples (1,218 total); the remaining original rows were held
out or not convertible to neutral property atoms. All 23 then-deployed tests passed
on the server; local tests subsequently passed 24 cases with two optional torch
tests skipped. A subsequent server run passed all 24 tests, including the alias
handling, multisource export and resampler gradient checks.

The 3B low-memory attempt subsequently failed during weight loading: the process
used 5.67 GiB when another 44 MiB allocation exhausted the available GPU memory.
It produced no optimizer steps. A separate Qwen2-VL-2B fallback preserves the
existing data and failed run, with its model source revision recorded at download.
The backend selects the matching Qwen2 or Qwen2.5 vision implementation; a 3B
adapter/resampler is never loaded into the 2B model.
The 2B source revision is `895c3a49bc3fa70a340399125c650a463535e71c`.
Its first run loaded successfully but ran out of memory on the first backward
pass (4.81 GiB process usage, 144 MiB allocation request, 113.56 MiB free). No
optimizer step completed. The retry uses the separately available GPU with about
7.8 GB free, requires at least 7 GB at launch, and writes `region_sft_2b_v1` and
`latent_reader_2b_v1` rather than overwriting the failed v0 run.
This retry progressed beyond 100 of 200 optimizer steps. Final metrics still
need to be read from its result files. The episode stage is queued after native
reader alignment: 32 optimizer steps on 24 examples (eight next-step, eight crop
read, eight region targets), then 64 alignment steps initialized from the existing
compact reader. These are supervised interface pilots, not completed paper training.
The run later remained at step 107 while its process stayed alive. Read-only
inspection reported a futex wait and high host I/O pressure (some avg10 94.55,
full avg10 86.93), alongside memory pressure. This is a verified resource wait,
not a completed training result; the existing trainer and dependent queue are
preserved. No other project's processes were stopped or modified.

Native GQA scene graphs were acquired and matched to program object IDs for all
32 source questions. Their region candidates are published as
`efa/viscot_gqa_native_v2`, HF commit
`82b5c2f4d6f7854cd44aa153342eda41955d3d62`. On the first eight tasks the added
candidates yielded one teacher-approved multistep episode (dev), compared with
zero for that subset under the preceding prompt. This is a small annotation
comparison, not evidence of student-model improvement.

The resource wait subsequently cleared. The 2B region run completed 200 optimizer
steps: training loss 0.793738, runtime 1,028.89 s. Positive compact-reader alignment
completed 256 updates on 250 eligible training region tasks in 93.99 s. The episode
run then completed 32 updates on the 24 exported interface examples: loss 0.748369,
runtime 11.15 s, followed by 64 warm-start reader alignment updates in 25.38 s.
These losses measure training fit, not held-out performance. The episode export
contains only three distinct multistep training trajectories.

The new `efa.evaluate` compares native and eight-token crop readers using the same
neutral queries and native boxes on hash-selected held-out ZwZ images. It also
runs direct full-image answers and the full bounded EFA controller on two cases.
The source boxes are oracle inputs only for the component comparison; the full
controller must route its own evidence. Cross-image source features and parent
context are reset between examples. Gate thresholds remain uncalibrated.

The first measured held-out run completed in 126.82 s on 16 records / 14 distinct
images. Strict canonical exact match was 1/16 for each reader; mean reference NLL
was 1.72947 native and 1.47074 compact. These are poor results, and the lower NLL
does not establish improved answer quality. The literal reference
`rectangular/rectangle` also illustrates a known strict-match limitation; original
outputs and reference strings are retained without retrospective score changes.
Direct whole-image answering was 0/2; EFA was 1/2, with 0/2 verified. Both EFA cases
took one step, three repair rounds and ten audits. The helmet case had invalid
routing throughout and an incorrect fallback; the flexible-pipe material case had
valid regions but failed binding throughout, changing plastic to the correct metal
only through unverified common-context fallback. Neither is a successful verified
repair. These two cases cannot validate multistep reasoning or performance gains.

The result bundle, policy, checkpoint hashes and training metrics are published as
`efa/heldout_2b_v0` in QC1/FBC, commit
`0ee7f9c06870e1c0d5b05b41b6643dd2b8dcdfa3`. All 25 current tests passed on the server;
local tests passed with the two optional torch tests skipped. A fixed-case stage
diagnostic now compares the original model, region adapter and episode adapter
under neutral and original-question crop prompts to identify the reader failure.

The stage diagnostic completed: base/region/episode native crop exact matches were
0/1/1 with the neutral interface and 2/2/2 with the original question. This small
comparison does not implicate episode SFT as the main cause. Upstream ZwZ also
deliberately filters for questions that Qwen3-VL-8B finds difficult; its source card
does not claim these are easy foundational region-reading examples.
Diagnostic outputs are published at `efa/reader_diagnostic_2b_v0`, HF commit
`284f16026178e85ae34589ca2091ba9d81569054`.

Source inspection exposed a concrete alignment concern in `zwz-21434`: the
unchanged parquet question refers to a garment on the right, its answer is purple,
but its pixel box `[171.3,257.1,556.2,1159.6]` lies on the white garment at left in
the linked original 1600x1600 image. The right garment is pink and the foreground
garment purple. The public synthesis code uses pixel xyxy boxes and overlays red
rectangles on original images; it provides no basis for an arbitrary coordinate
rescaling or answer replacement. This case needs source adjudication. Keep the
published pilot and its literal metrics as diagnostic artifacts, not a validated
model ranking. Do not expand the existing unreviewed ZwZ conversion into formal
training. The next data stage must use source-checked original-image bindings,
including GQA scene-graph referents, before training/evaluating the reader again.

Sources inspected: ZwZ dataset card at revision
`fec404cb7bd5a18458e5add2802a5fb183981d14`, parquet row 21434, and
`inclusionAI/Zooming-without-Zooming/data_synthesis/create_vqa.py` (main).

The offline approvals are teacher pseudo-labels. No calibrated
threshold, verified coverage, benchmark accuracy, successful repair rate, or
paper-level improvement is claimed. Full reader discrimination objectives,
student error/repair trajectories, revision detection, and matched evaluations
remain required. Native GQA scene graph candidates improve the region supervision
source without inventing intermediate labels.
