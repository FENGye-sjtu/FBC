# Training-free CVPR2027 track

The user's latest 2026-09-21 instructions supersede the older root research
contract **for this directory only**. Start from FaithAct; freeze the reasoning
MLLM; generate one step, verify, repair/reject, then continue. Add localized
counterfactual verification. Do not switch this track to DeepEyes SFT.

All new server files/caches belong exclusively to `/data_cinema/zqc/CVPR` or
`/data_wyn/zqc/CVPR`. Existing VBC resources may be read, never modified. Preserve
other GPU jobs and local uncommitted work. Sync verified code to origin/main
without force-pushing; no credentials or large data in Git.

Prioritize runnable FaithAct reproduction and end-task accuracy. Record actual
measurements, distinguish small engineering runs from paper reproduction, and
never describe model verifier acceptance as guaranteed correctness.
