# FBC / Evidence-First Auditing

Implementation of **Audit Before Advancing**, following the user's
`VBC_CVPR2027_EvidenceFirst_v2_Full.pdf`. The manuscript's empirical tables are
illustrations, not measured results. See [research contract](docs/RESEARCH_CONTRACT.md).

## Current implementation

- Numbered step generation, value-blind region acquisition, incumbent-only B/S/N/D
  audit, at most 3 proposals in each of 3 failure-triggered review rounds.
- Common-context fallback explicitly remains unverified. A versioned ledger
  inherits all actually exposed parents and supports descendant revocation.
- Executable Qwen2.5-VL backend with original-image crops, teacher-forced scores,
  source-coordinate/version logs and a separate premise-only deduction path.
- Native ZwZ region QA conversion, offline teacher episode annotation, and
  DeepEyesV2/LLaMA-Factory SFT export with a 3B four-bit LoRA configuration.

The native-crop backend is an engineering baseline. An experimental eight-token
reader is integrated and has completed a 40-step positive-region alignment smoke.
The step interface supports parent re-audits and regenerates after revocation;
reliable detection of when to revisit still requires training/evaluation.
Initial thresholds are uncalibrated. No benchmark
accuracy, efficiency gain, or full EFA training completion is claimed.

## Setup and data

```sh
python -m pip install -e '.[data]'
python scripts/fetch_upstream.py
python -m unittest discover -s tests -v
python -m efa.zwz --metadata data/raw_metadata/inclusionAI__ZwZ-RL-VQA/train.parquet --output data/zwz_seed_v0 --limit 64
python -m efa.export --root data/zwz_seed_v0 --output data/sft_v0 --engineering-only
```

`configs/upstream.json` pins the inspected DeepEyesV2 source. Its cold-start recipe
uses LLaMA-Factory; `configs/sft_3b.yaml` adapts that path to 3B QLoRA. Upstream
source is fetched reproducibly and retains its own license, rather than being
silently copied into this repository.

Raw QA, region atoms, reviewed complete episodes, and student repair rollouts are
counted separately. An existing bounding box does not establish a multi-step
explanation. Pixel groups are split before variants. Benchmark and near-duplicate
clearance is still required for formal training; the engineering override is
recorded in each export manifest.

The offline annotation command is `python -m efa.annotate --root DATA --limit 32`.
It needs `EFA_TEACHER_BASE_URL`, `EFA_TEACHER_MODEL`, and `EFA_TEACHER_API_KEY` in
the environment. No provider address is guessed from a key. Teacher output stays
pseudo-labeled until independently checked; no artificial runtime PASS or score
labels are generated.

## Server

All new remote artifacts stay in `/data_cinema/zqc/VBC` or `/data_wyn/zqc/VBC`.
`scripts/server_env.sh` scopes caches and temporary files. `scripts/remote.py`
supports deployment, project-relative upload and shell-script execution with
interactive password input. It does not persist credentials.

Run `scripts/setup_server.sh`, `scripts/setup_model.sh`, then
`scripts/setup_sft.sh` on the server. Upload region data to
`data/derived/zwz_seed_v0` and run `scripts/run_sft.sh` for a ten-step reader smoke.
This validates the training path, not scientific effectiveness. Other GPU jobs
must remain untouched.

Code is synchronized to GitHub `FENGye-sjtu/FBC`; data to the private Hugging Face
dataset `QC1/FBC`. Pull main before editing and push verified commits without
force so another machine can continue the work.
