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

The VLM generates one structured visual claim or deduction and, optionally, a
concise answer. A visual step must include a nonempty neutral localization query.
An empty or whitespace-only visual query is deterministically replaced with the
public question stem after removing answer choices and answer-only instructions.
This supplies a stable evidence-budget key without consuming a repair attempt
as a formatting error. A logical deduction uses an empty localization query:
it combines ledger premises rather than requesting a new image observation.
A visual claim may include an answer;
the controller audits both its image support and its answer-to-question mapping.
This avoids conflating a formatting choice with lack of evidence. Coordinate
generation is delegated to a separate localization call; internal boxes are not
fed back as output examples.

If the visual assertion passes but its proposed final answer is not entailed,
only the verified assertion is committed. The answer is withheld and must be
derived again at the next step. We do not discard a verified observation merely
because its attached conclusion is invalid, nor return that invalid conclusion.

For multiple-choice questions, an additional answer-consistency check scores all
option labels in a fresh image context without the candidate claim or answer.
An attached answer is withheld when another option has higher frozen-model
likelihood. This is independent answer rescoring, not an image counterfactual
intervention or proof that the higher-scoring option is correct. It uses no
reference label and no learned auxiliary verifier. Single-token option scores
share one prefix forward pass and retain full-vocabulary normalization;
multi-token options use complete teacher-forced sequence likelihoods. This check
does not itself retract the accepted visual claim.

## Visual support and counterfactual specificity

Let p(c; I) be the normalized Yes-vs-No likelihood of a fresh verification query.
This is a decision score, not a calibrated probability of correctness. Initial
support requires p(c; I) >= tau_s. When a repaired candidate has a localized
region, the verifier sees the full image and its crop under a fixed total
visual-token budget. A full-image-only repair uses the complete image.

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
attempt. The current repair variant obtains a fresh evidence reading without
showing the rejected claim, its proposed answer or assertion-bearing rejection
feedback to the acquisition and reading calls. On the first repair, the neutral
public question stem guides independent localization; the reader inspects the
full image and a crop with a small context margin. On the second repair, the
reader's neutral question about a missing visible cue guides a new localization;
if no missing-cue question was produced, the public question is reused. The third
repair reads the complete image to recover context or coverage that a crop may
have missed.

Each reading returns directly visible observations, a remaining evidence
question and an evidence-completeness flag. These are tentative model outputs.
The completeness flag never grants verification status. A separate proposal
call uses this tentative reading to form a replacement atomic step, which must
pass the ordinary audit before becoming an established premise. Reading and
replacement occur inside the existing repair round and do not create extra
repair budgets. Successful initial steps do not incur a rereading call.

If all three repairs remain inconclusive or contradicted, the controller commits
an explicitly unresolved proposition ("uncertain whether c"), not c as a fact.
The raw tentative proposition is retained in the trace, while downstream prompts
expose its unresolved evidence question rather than repeatedly asserting the
unverified value. The controller records the normalized evidence questions used
within an exhausted repair cycle. An exact normalized repeat is not audited and
receives no new repair budget, even if the proposed kind changes from visual to
deduction. An identical already-verified visual claim with the same normalized
question is also not re-audited, including when its attached answer was withheld.
The controller requests a new evidence point or a logical deduction with an
empty evidence question. Two ignored redirections trigger uncertain finalization;
a successful redirect continues the ordinary step budget.

These guards use exact normalization, not semantic identity. Paraphrased
questions can still refer to the same underlying evidence point; the current
implementation does not claim to eliminate that loophole. Empty-query fallback
groups unspecified visual queries under the public task stem, without claiming
that all observations about that task are semantically identical.

An unresolved point does not prevent a subsequent conditional deduction.
Deductions inherit uncertainty from exposed uncertain premises; the current
implementation conservatively records all exposed ledger entries as parents.
Independent visual facts can still receive their own support judgment, while
final chain status remains uncertainty-aware.

When finalization is necessary and no ledger fact is verified, the current repair
variant selects the valid audited visual candidate answer with the highest
unmasked support score. Exact score ties select the latest candidate. Tentative
candidates remain outside the verified ledger, and the selected answer is
explicitly uncertain. This rule can retain a repaired hypothesis instead of
automatically restoring the initial answer. Support scores across different
claims and image views are not calibrated correctness probabilities, so this
selection rule requires empirical evaluation. The selected claim, score and
selection source are recorded. If no valid scored visual candidate exists, the
inherited initial-answer or best-effort synthesis fallback remains available.
If verified facts exist, the VLM synthesizes from the labeled ledger. Such budget
or redirection finalization is always reported as uncertain, not as having passed
a final audit. All questions remain in the accuracy denominator.

Earlier engineering implementations used progressively enlarged crops and, when
no evidence was verified, retained the initial answer. Their traces remain
separate comparisons; they are not relabeled as outputs of the independent
rereading and support-ranked fallback described above.

The V16 forward-only controller also closes all normalized question keys actually
encountered in a successfully repaired point. Rephrasing its claim or changing
its kind cannot reopen that same key at a new step. Earlier failed variants are
closed, not labeled verified; the ledger retains only the claim that actually
passed. Independent evidence and empty-query logical deductions can continue.
This extends the earlier exhausted-point and exact-claim guards; it does not
identify unseen semantic paraphrases. V15 and earlier traces retain their
original controller behavior and must not be relabeled as V16 results.

Dependency-specific backtracking is future work, not a claimed implemented feature.

## Implementation and evaluation commitments

One frozen Qwen2.5-VL instance performs proposal, localization, support scoring
and repair. No verifier training, segmentation network or depth estimator is
required for our method. The FaithAct baseline separately uses its official
pretrained CLIP polling checkpoint and GroundingDINO, with pinned source and
explicitly documented reproduction differences.

Compare matched CoT, FaithAct port, stepwise support and stepwise CF settings.
Primary outcome is answer accuracy over all cases; also report uncertainty rate,
repair rounds, effective audits per evidence point, calls, generated/input tokens
and wall time. Also report accepted-answer correctness, initial-to-final
corrections and regressions, and the contribution of uncertain finalization.
NF4 engineering runs
must not be called BF16 paper reproduction. Keep the early engineering cases
identified and separate from the later frozen-protocol evaluation report.

### Implemented development variant: atomic_v1

The optional `atomic_v1` repair variant limits the independent reading's
observation to 180 characters and the repaired claim to 140 characters, using
separate schemas. Its prompts target one checkable observation answering the
current neutral evidence question and discourage copying background description
or compound explanations into the claim. These length limits do not guarantee
semantic atomicity or correctness. The variant adds no model calls and leaves
support checks, counterfactual checks, repair budgets and uncertainty handling
unchanged. Claim length and lexical truncation signals are logged for diagnosis,
without affecting acceptance. The default `reread` variant retains its existing
prompts, schemas and behavior for matched comparisons. No performance improvement
is claimed from this implementation alone.

## Results placeholder

No improvement claim is inserted here until supported by completed matched runs.
Completed BF16 reproduction and prompt-validation results are tracked in
`STATUS.md`. The current V15 online method is 9/16 on an NF4 development subset,
versus matched CoT 10/16; the queued BF16 online evaluation has no result yet.
Neither the early four-case pilots nor these 16-case diagnostics establish
benchmark-level superiority.
