# Current experiment handoff

SSH access recovered on 19 September 2026. The previous v2 pipeline completed;
it was not restarted. All server paths below are relative to
/data_cinema/zqc/VBC. Preserve unrelated projects and GPU jobs.

## Completed baseline

- runs/gqa_sft_2b_v2: 1,200 SFT updates, loss 0.179668, runtime 958.665 s.
  Actual visual modules remain BF16; language modules include NF4 weights.
- runs/latent_gqa_2b_v2: 768 positive alignment updates on 384 atoms from
  126 train episodes, runtime 205.483 s, peak allocated memory 3,021,314,560 bytes.
- runs/gqa_heldout_2b_v2: all three dev cases generate three steps. Strict
  canonical string scores: native reads 8/10, compact reads 6/10, direct QA 2/3,
  unaudited steps 1/3, full EFA 1/3; zero effectively verified final outputs.
- The first direct answer is "in car" against reference "car": a literal scoring
  mismatch, not proof of a semantic perception error. The full EFA trace also
  shows a genuine harmful change: a correct incumbent "cat" failed binding and
  eventually fell back to "bear". Relation orientation fails as well.
- Traces, training configuration and precision records are published as
  efa/gqa_heldout_2b_v2, HF commit 70965b097aa5add2086da25b5bb54138f68b5688.

## Current next run

The v0 reader-objective attempt exited with CUDA OOM when GPU 0 had competing
allocations (16.06 GiB and 4.39 GiB from other processes; this process used
3.18 GiB). Its logs and partial losses are preserved. No other job was modified.
scripts/train_reader_objectives_2b.sh now submits the unchanged recipe to GPU 1
with log label reader_objectives_2b_v1. Check actual progress with
scripts/status_reader_objectives.sh before launching anything else.

It retains the v2 shared adapter, initializes its reader from v2, and trains
summed read NLL, identity NLL and whole-sequence negative unlikelihood for 768
updates. This changes a bundle of objectives, not only the negative term; an
improvement cannot be attributed to one component without later ablation.
Outputs are runs/reader_objectives_2b_v1 and runs/gqa_reader_objectives_eval_v1.
The same three dev cases and uncalibrated policy are used. Completion must be
read from artifacts, not inferred from the launch.

All 45 tests passed on the server after the latest code deployment. Both
data/derived/gqa_reader_objectives_v1 and data/derived/clevr_count_program_v1
are uploaded. The mixed GQA+CLEVR SFT launcher is ready but has not been
submitted; avoid competing for the same free GPU while the reader run is live.

## Data and outstanding research

- Reviewed GQA: 156 approvals from 512 candidates; 126 train / 11 dev /
  19 calibration. HF commit e19bda2b491706c57cabe7873767c7258eb6311a.
- Reader objectives: 384 train records, 54 native category negatives on 27 atoms.
  HF commit 6114f79c0c3b3b521c63cf1140b11ad26325e2f9.
- CLEVR: 256 native three-step episodes; 213 train / 27 dev / 16 calibration.
  HF commit 3d694fb95e7a7f52de40e2e0c3a59a19ac04fa40.
- The validated mixed export has 2,856 SFT examples, including 213 image-free
  premise reads. See CLEVR_EPISODES.md for source policy and limits.

No current result establishes a gate or repair advantage. Still required:
reliable learned routing/reading, predicted/incomplete sources, actual student
repair and revisit training, near-duplicate/benchmark clearance, calibration,
matched comparisons and paper-scale measured results. The CVPR goal remains open.
