# Handoff — 2026-09-21

This is an implemented and deployed **first engineering version**, not a
completed paper or a successful reproduction claim.

## Verified now

- Code is isolated in `trainingfree/`, synced to `FENGye-sjtu/FBC:main` and
  deployed to `/data_cinema/zqc/CVPR/code/trainingfree`.
- 12 CPU unit tests pass locally and on the server (controller, repair context,
  counters, image interventions and matched-control gate logic).
- Real Qwen2.5-VL-3B direct generation ran on CPU. First online smoke exposed
  malformed step JSON; schema-constrained generation was added and the second
  real-model smoke was launched. Do not treat these one-case tests as accuracy
  evidence or claim the second run passed before reading its output.
- Official FaithAct `best.pt` downloaded and SHA256 verified. GroundingDINO and
  CLIP architecture/tokenizer files are available. The full classifier checkpoint
  contains CLIP encoders; strict state loading is exercised by the tool smoke.
- GitHub push succeeded. HF authenticated identity is QC1; QC1/FBC is accessible
  and private. No fabricated benchmark artifacts were published.

## Running / queued at handoff preparation

Read live files instead of assuming these transient states remain unchanged:

| Job | PID at launch | Live record |
|---|---:|---|
| Asset preparation / pretrained 7B weight hash verification | 3678436 | `runtime/assets.json`, `logs/prepare_v3.log` |
| CPU real-model structured-step smoke | 3674804 | `logs/cpu_adapter_smoke_v2.log`, `runs/cpu_adapter_smoke_v2/` |
| CPU real FaithAct tool smoke | 3678438 | `logs/tools_smoke.log`, `runtime/tools_smoke.json` |
| 16-case paired pilot, then 765-case baseline | 3678437 | `runtime/faithact_reproduction_v1.json`, `logs/faithact_reproduction_v1.log` |

All paths in this table are relative to `/data_cinema/zqc/CVPR`.
The queue waits for complete assets and a GPU with >=22 GB free and <=5% usage;
it does not evict other jobs. Four GPUs were occupied. It times out after 48h.
Only this track's own asset/queue processes were restarted during implementation.
Older VBC/covt files and all pre-existing GPU tasks were left unchanged.

## Next work

1. Inspect the two real smoke outputs and fix any adapter/runtime errors first.
2. Confirm all five asset entries are ready. The 7B source is an existing official
   snapshot, read-only and checked against pinned HF LFS hashes. RealWorldQA must
   contain exactly 765 records.
3. Read the paired pilot table and traces. Improve actual failure modes, not a
   large collection of peripheral experiments. Run full baseline reproduction and
   record deviations from paper protocol before comparing its reported scores.
4. Publish measured traces/summaries to `QC1/FBC/trainingfree/` without changing
   older datasets; keep code and coordination documents on GitHub main.

Fixed design and experiment priorities: `RESEARCH_PLAN.md`. Operational commands
and reproduction caveats: `README.md`. Do not switch this task back to SFT.
