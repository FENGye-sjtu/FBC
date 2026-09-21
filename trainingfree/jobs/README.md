# Experiment requests

These JSON files are versioned requests, not instructions to launch every file.
Older requests are retained for provenance; some intentionally reproduce fixed
failures. Inspect STATUS.md and the server worker result before submitting.

Current sequence on 2026-09-22:

- BF16: `7b_bf16_replication_v3.json`, full 765-case CoT/FaithAct reconstruction.
  Versions v1/v2 failed placement and are historical, not fallback launch choices.
- NF4: completed `7b_dev_002_pilot4_v3.json`, then v4 controller check,
  `7b_dev_003a_repair_regression_v5.json`, then
  `7b_dev_004_pilot16_v5.json` (declared standardized FaithAct answer format).
- 3B and the old full NF4 baseline are superseded diagnostics.

Submit to the appropriate resident worker inbox using a unique filename and
atomic rename from a non-JSON temporary filename. Never overwrite an active
request, reuse an output directory with changed code/config, or assume queue
process exit zero means the experiment succeeded: inspect worker results and
per-case outputs. Do not create another model instance if the matching resident
worker can run the request.
