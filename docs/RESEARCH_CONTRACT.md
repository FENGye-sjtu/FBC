# EFA implementation contract

The user's manuscript `VBC_CVPR2027_EvidenceFirst_v2_Full.pdf` (18 September 2026)
defines the story and architecture. The framework must generate numbered steps,
pause at each boundary, associate every required atom with original-image evidence,
audit the incumbent before proposing alternatives, and advance only with recorded
audit/provenance status. Implementation details may change through experiments.

## Fixed method

- DeepEyesV2 is the engineering base, pinned in `configs/upstream.json`.
- One deployed VLM performs generation, neutral routing, source reading and repair.
- Acquisition excludes the incumbent and the main reasoning cache. Existing values
  appear only as teacher-forced scoring suffixes.
- Initial audit uses B/S/N/D; it never decodes alternative semantic answers.
- The controller outputs PASS/REVIEW. Failure allows up to three rounds, three
  canonical proposals per round, each re-audited with the same gate.
- Exhaustion rescores all observed values under one common source context and
  continues explicitly unverified. No fallback is labeled PASS.
- Actual exposed parents determine inherited status. Changed roots revoke descendants.
- Regions can be source sets. Deduction checks explicit premises and inherits their
  original-image provenance. A single arbitrary box cannot certify a deduction.
- Use SFT first and a small model when memory is constrained. RL is not a prerequisite.

## Data

Reuse DeepEyes/DeepEyesV2, ZwZ and Visual-CoT first, supplement native GQA/VG/program
annotations as needed. A region QA record is not yet a complete EFA episode. Count
raw QA, region atoms, complete episodes and actual student repairs separately.
Never fabricate B/S/N/D scores, model repair traces, benchmark numbers or labels.
Teacher labels are distinguished from native annotations and independent adjudication.
Split by image group before variants; clear overlap with final benchmarks before
formal training. Small engineering-only exports are explicitly marked.

## Server and synchronization

All new server files, caches, environments, models and outputs must be under
`/data_cinema/zqc/VBC` or `/data_wyn/zqc/VBC`. Do not modify other project files,
kill other jobs, or use their mutable caches. The result disk is almost full;
place heavy artifacts on the cinema volume. Credentials stay out of all tracked files.
Before code changes fetch main; after complete verified changes commit and push
to FENGye-sjtu/FBC without force. Upload data versions to QC1/FBC, official endpoint.

## Current implementation tiers

`efa.core` implements the bounded resolver and versioned ledger. `efa.qwen` is the
first executable real-model backend using independently encoded native crop tokens.
It is a baseline until the trained eight-token reader is integrated and validated.
`efa.latent.SourceResampler` defines the trainable eight-token compression module;
random weights are not a usable verifier. No efficiency or calibration claim is made.
Automatic late-error localization and complete rollback/regeneration orchestration
still require integration beyond the ledger's tested graph revocation.

The initial LoRA smoke trains native region reading/routing only. It is not the
paper's full three-stage model. Full episodes require teacher annotation followed by
source checks, and repair episodes require actual student/controller runs.
