# Active experiment handoff

Last updated: 19 September 2026, 05:00 Asia/Shanghai (client clock).

The main research goal is not complete. The latest submitted pipeline is
`gqa_mixed_training_2b_v2`, launched through `scripts/train_gqa_mixed_2b.sh`.
It runs inside `/data_cinema/zqc/VBC`; do not restart it before checking whether
the existing process or its outputs are still active.

- Shell PID at launch: 606353; SFT Python PID last seen: 606583.
- Log: `logs/gqa_mixed_training_2b_v2.log` relative to the server VBC root.
- SFT output: `runs/gqa_sft_2b_v2`, 1,200 planned optimizer updates.
- Reader output: `runs/latent_gqa_2b_v2`, 768 planned positive alignment updates.
- Evaluation output: `runs/gqa_heldout_2b_v2`, three hash-selected dev images;
  direct answering, unaudited numbered steps, and full EFA use the same checkpoint.
- Last confirmed SFT progress: 862/1200, still active. Actual precision callback
  recorded BF16 vision (`Linear`) and NF4 language (`Linear4bit`). Neither SFT
  completion nor reader/evaluation outcomes have been observed yet.
- Monitoring then lost its SSH connection; a reconnect and TCP check timed out.
  This is an observation-access failure, not evidence that the detached job stopped.

Training data: `data/derived/gqa_native_review_v2_complete` on the server.
The 512 reviewed trajectories yield 156 approvals (126 train / 11 dev / 19
calibration), 475 total approved steps and 384 training steps. HF data commit:
`e19bda2b491706c57cabe7873767c7258eb6311a`, private `QC1/FBC`, subfolder
`efa/gqa_native_review_v2_complete`.

Use `scripts/status_gqa_training.sh` or `scripts/monitor_gqa_mixed.sh` through the
scoped remote helper to reconnect. After the evaluation summary exists, package
with `scripts/package_gqa_mixed_results.sh` and publish the completed artifact.
Preserve the v0 historical results and v1 failed launch directories. The v1
callback failure exposed an unrecognized LLaMA-Factory method: the corrected
configuration uses `bnb`, not `bitsandbytes`.

Remaining scientific work includes reliable negative/identity objectives,
predicted and incomplete source training, natural student repairs and revisits,
numerical/deductive data, benchmark decontamination, calibrated gates and matched
paper evaluations. The current 35 tests pass on the server; they do not establish
research effectiveness. No benchmark gain or submission-readiness claim is made.
