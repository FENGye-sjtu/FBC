# Training-free CVPR2027 track

The user's latest 2026-09-21 instructions supersede the older root research
contract **for this directory only**. Start from FaithAct; freeze the reasoning
MLLM; generate one step, verify, repair up to THREE rounds after the initial
attempt; if still unresolved, record explicit uncertainty and CONTINUE reasoning.
Never silently promote uncertainty to a verified premise. Add localized
counterfactual verification. Do not switch this track to DeepEyes SFT.

All new server files/caches belong exclusively to `/data_cinema/zqc/CVPR` or
`/data_wyn/zqc/CVPR`. Existing VBC resources may be read, never modified. Preserve
other GPU jobs and local uncommitted work. User's latest update: sync to GitHub/HF
only at meaningful research milestones, NOT after every edit. No force pushes,
credentials or large data in Git. Busy GPUs may be shared; ignore utilization and
schedule according to the actual memory requirement, using NF4 or smaller models
for engineering runs when needed. Label quantized runs separately from BF16 replication.

Prioritize runnable FaithAct reproduction and end-task accuracy. Record actual
measurements, distinguish small engineering runs from paper reproduction, and
never describe model verifier acceptance as guaranteed correctness.
