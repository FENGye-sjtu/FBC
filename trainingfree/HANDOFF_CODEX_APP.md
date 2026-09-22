# Codex App handoff (2026-09-23)

Task: continue the training-free FaithAct reproduction and step-boundary
Evidence First track in `trainingfree/`, respecting `trainingfree/AGENTS.md`.
Do not silently resume the older `efa/` SFT track or overwrite its dirty tree.
This file is a handoff for the other Codex conversation; it does not send a
message to that conversation automatically.

## Established results

- Released FaithAct contains `TEMPLATE_GCOT`, but the missing Qwen initial
  generation adapter and RealWorldQA Table-2 scoring path prevent exact paper
  parity. See `reports/20260922_faithact_prompt_protocol_audit.md`.
- On first-128 development IDs, paper-literal CoT 94/128 becomes 87/128 after
  released-code FaithAct refinement. On the full 765 layout-v5 reconstruction,
  CoT 542/765 becomes FaithAct 510/765 under the same parser. Neither is the
  claimed 74.5% paper reproduction.
- V20 NF4 first-16 Evidence First and image-CF both end 10/16, exactly their
  initial paper-literal CoT anchor, with zero corrections and zero regressions.
  See `reports/20260923_v20_repair_guard_diagnostic.md` for hashes and traces.
- V21 guards against laundering a repaired deduction into a visual fact and
  replaying an identical accepted claim under a different evidence question.
  Its performance is unmeasured until its independent run completes.

## Live work and checks

Connect with `ssh sjtu-server`, which selects the configured key; direct SSH to
the IP did not select it. All server writes stay under `/data_cinema/zqc/CVPR`
or `/data_wyn/zqc/CVPR`. No credentials belong in Git or reports.

1. Read `/data_cinema/zqc/CVPR/runtime/queues/v21_repair_guard/results/job001.json`
   and `/data_cinema/zqc/CVPR/logs/v21_repair_guard16.log`. Count both JSONL
   output files in `runs/realworldqa_vlmevalkit_7b_nf4_pilot16_v21_repair_guard`.
   The job uses an immutable code snapshot under `code_snapshots/v21_repair_guard`
   and holds the GPU-2 project lock only while running.
2. When V21 has 16/16 records in both arms, run `trainingfree.analyze_online`
   with the public `data/vlmevalkit/cases.jsonl` and compare each arm to V20 on
   the exact same IDs, parser and budget. Check repeated-step counts, false
   model-verified answers, uncertain finals, errors, latency and CF mask calls.
   Do not claim a gain from the initial CoT anchor or a parser-only recovery.
3. The separate BF16 queue `runtime/queues/layout_v5_bf16` was running job004
   (V16 online, IDs 128-255) when this handoff was written. Job005 is V20
   Evidence First on those IDs and was pending. Inspect result JSON files and
   actual row counts before reporting a score. Do not reload or replace its
   mutable shared code directory while this queue is active.
4. If V21 fixes the guard but does not improve accuracy, inspect the paired
   failures before changing prompts or thresholds. Freeze each candidate and
   report raw outputs, actual reference-blind scores and the original FaithAct
   protocol gap. Keep development labels separate from held-out testing.

GitHub canonical branch is `FENGye-sjtu/FBC` `main`. On this workstation,
`github.com:443` was unreachable directly; a temporary SSH SOCKS tunnel via
`sjtu-server` allowed `git -c http.proxy=socks5h://127.0.0.1:18111 fetch origin
main`. The tunnel is transient, not a stored credential. Fetch/rebase before
future pushes; never force-push and never commit raw datasets or secrets. Update
this handoff with measured V21/BF16 outcomes before the next research milestone.
