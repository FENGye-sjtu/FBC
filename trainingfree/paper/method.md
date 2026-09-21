# Method draft: Evidence-gated visual reasoning with bounded repair

Status: implementation-aligned draft, not a completed empirical claim. Working
title only. The reasoning VLM is frozen. All reported results must identify model,
precision, image/token budget, benchmark split and method version.

## Motivation and scope

A wrong visual premise can become part of the context used for later deductions.
Our objective is to audit a candidate before exposing it as an established premise,
and to repair unreliable evidence locally rather than regenerate an entire chain
after completion. We do not claim that all prior work is post-hoc or that an
accepted model judgment guarantees truth. Related online-verification methods
must be included in the final related-work comparison.

The framework distinguishes verified-by-model evidence from unresolved evidence.
After three unsuccessful repair rounds, it continues with an explicit uncertainty
label instead of silently accepting the claim or abandoning the question. This
bounded continuation is part of the method, not an exception hidden from metrics.

## State and step generation

At step t, the ledger contains claims c_i, evidence regions R_i and epistemic
labels z_i in {verified, uncertain}. Candidate text is tentative until audited.
Only this labeled ledger, the question and image are supplied to the next step;
rejected drafts are not copied into the committed reasoning history. A repair
instruction is temporary context at the current boundary.

The VLM generates one structured visual claim or deduction, a neutral localization
query and, optionally, a concise answer. A visual claim may include an answer;
the controller audits both its image support and its answer-to-question mapping.
This avoids conflating a formatting choice with lack of evidence. Coordinate
generation is delegated to a separate localization call; internal boxes are not
fed back as output examples.

If the visual assertion passes but its proposed final answer is not entailed,
only the verified assertion is committed. The answer is withheld and must be
derived again at the next step. We do not discard a verified observation merely
because its attached conclusion is invalid, nor return that invalid conclusion.

## Visual support and counterfactual specificity

Let p(c; I) be the normalized Yes-vs-No likelihood of a fresh verification query.
This is a decision score, not a calibrated probability of correctness. Initial
support requires p(c; I) >= tau_s. On repair, the verifier sees the full image and
a zoomed evidence region under a fixed total visual-token budget.

For a localized region R, deterministic mean-color masking constructs I_minus_R.
Two equal-area, low-overlap control regions C_k quantify nonspecific occlusion
effects. The counterfactual specificity score is

    S(c,R) = [p(c;I) - p(c;I_minus_R)]
             - mean_k [p(c;I) - p(c;I_minus_C_k)].

The CF gate requires original-image support and S(c,R) >= tau_cf. Dependence on
a region alone is not evidence of correctness. Missing localization or unusable
controls produces an inconclusive audit, triggering repair rather than a pass.
If the verifier uses a zoomed view, it is regenerated from each intervened image:
the unmasked crop is never retained as a leak of deleted evidence.

Current engineering defaults are tau_s=0.8, tau_cf=0.05 and two controls. They
have not been claimed optimal or calibrated. Mean-color pixel masking is a
lightweight intervention proxy, not a claim of identifiable causal correctness.

## Repair and uncertainty-preserving continuation

An unsuccessful audit triggers up to three repair rounds after the initial
attempt. Each repair independently reacquires a region, enlarges its context
with increasing padding, and regenerates the current claim from the original
image plus the enlarged crop. Every replacement is re-audited.

If all three repairs remain inconclusive or contradicted, the controller commits
an explicitly unresolved proposition ("uncertain whether c"), not c as a fact.
The raw tentative proposition is retained in the trace, while downstream prompts
expose its unresolved evidence question rather than repeatedly asserting the
unverified value. If the next step merely repeats the exhausted claim without
new committed evidence, the controller continues to uncertain answer synthesis
instead of granting the same point a new three-round budget.
Deductions inherit uncertainty
from exposed uncertain premises; the current implementation conservatively records
all exposed ledger entries as parents. Independent visual facts can still receive
their own support judgment, while final chain status remains uncertainty-aware.

At a bounded step budget, the VLM returns the best-supported answer available
from the image and labeled ledger. Such an answer is reported as uncertain, not
as having passed a final audit. All questions remain in the accuracy denominator.
Dependency-specific backtracking is future work, not a claimed implemented feature.

## Implementation and evaluation commitments

One frozen Qwen2.5-VL instance performs proposal, localization, support scoring
and repair. No verifier training, segmentation network or depth estimator is
required for our method. The FaithAct baseline separately uses its official
pretrained CLIP polling checkpoint and GroundingDINO, with pinned source and
explicitly documented reproduction differences.

Compare matched CoT, FaithAct port, stepwise support and stepwise CF settings.
Primary outcome is answer accuracy over all cases; also report uncertainty rate,
repair rounds, calls, generated/input tokens and wall time. NF4 engineering runs
must not be called BF16 paper reproduction. Keep the early engineering cases
identified and separate from the later frozen-protocol evaluation report.

## Results placeholder

No improvement claim is inserted here until supported by completed matched runs.
The first four-case baseline pilot and its parser correction are engineering
diagnostics only; they cannot establish benchmark-level superiority.
