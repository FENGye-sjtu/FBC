# Repository Workflow

- Treat `origin/main` at `https://github.com/FENGye-sjtu/FBC` as the canonical code branch.
- Before changing code, fetch `origin` and start from the latest `main` so work from other machines is included.
- After a code change is complete and verified, commit it, rebase on the latest `origin/main`, and push it to `origin main`. Never force-push shared work.
- Store dataset artifacts in the Hugging Face dataset repository `QC1/FBC`, revision `main`, rather than in this Git repository.
- Use `HF_ENDPOINT=https://hf-mirror.com` for Hugging Face downloads. Use the explicit official endpoint `HF_ENDPOINT=https://huggingface.co` for dataset uploads.
- Never commit access tokens, credential files, generated secrets, or authenticated remote URLs.

## Research scope

- Follow `docs/RESEARCH_CONTRACT.md` and the user's EvidenceFirst v2 manuscript:
  step-boundary audits, value-blind acquisition, failure-only repair, and honest
  inherited verification status are fixed. Implementation choices may evolve.
- Use the pinned DeepEyesV2 engineering base and start with SFT/small models.
- All new remote files and caches belong under `/data_cinema/zqc/VBC` or
  `/data_wyn/zqc/VBC`. Preserve other projects and running GPU jobs.
- User authorizes project implementation, training and GitHub/HF synchronization.
  Never invent experimental results or call teacher pseudo-labels measured audits.
