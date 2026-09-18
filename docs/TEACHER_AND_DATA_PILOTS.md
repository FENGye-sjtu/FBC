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

Native GQA scene graphs were acquired and matched to program object IDs for all
32 source questions. Their region candidates are published as
`efa/viscot_gqa_native_v2`, HF commit
`82b5c2f4d6f7854cd44aa153342eda41955d3d62`. On the first eight tasks the added
candidates yielded one teacher-approved multistep episode (dev), compared with
zero for that subset under the preceding prompt. This is a small annotation
comparison, not evidence of student-model improvement.

The offline approvals are teacher pseudo-labels. No runtime PASS, calibrated
threshold, verified coverage, benchmark accuracy, successful repair rate, or
paper-level improvement is claimed. Full reader discrimination objectives,
student error/repair trajectories, revision detection, and matched evaluations
remain required. Native GQA scene graph candidates improve the region supervision
source without inventing intermediate labels.
