# Compact initial visual proposal: implemented and measured

Status: implemented behind `--initial-style compact_v1`; both V14 arms completed
after V13. V14 online scored 8/16 and V14 CF 7/16, equal to their respective
V13 controls. No final-accuracy gain was produced by this proposal. The original
pre-run rationale and planned comparison below are retained for provenance.
A better initial proposal is a hypothesis,
not an established explanation of the current accuracy gap.

## What the existing trace establishes

The cached high-resolution NF4 V9 CF pilot has 16 cases, 8 final correct,
14 valid initial option answers, and 7 correct initial answers. Report both
7/14 conditional initial accuracy and 7/16 correct initial answers over all
cases; the other two cases are not initial errors eligible for a correction
claim. On the 14 comparable cases, initial and final answers have no correctness
gains or losses. The same-case high-resolution NF4 controls previously recorded
9/16 direct and 10/16 CoT correct. These small counts motivate inspection, not a
conclusion that a compact prompt will improve performance.

There are three specific first-proposal issues in this cached trace:

- Case 0 is labeled `kind="deduction"` despite an empty ledger. The controller
  necessarily rejects it with `A deduction requires verified premises.` Its
  claim includes new image observations, so it belongs in the visual audit path.
- Case 7 is already `kind="visual"`, but `final_answer="N"` is not a public MCQ
  option. It is an invalid answer representation, not a missing visual step.
- Case 9 is also `kind="visual"`, with the literal string
  `final_answer="null"`, rather than JSON null. Its claim says the relevant
  distance is unresolved. Genuine null is appropriate when the evidence is
  insufficient; it should not be forced into an option.

Thus, choosing a deduction on an empty ledger creates one avoidable first-step
rejection, but does **not** explain the two unavailable initial answers. Do not
conflate the kind error, invalid option formatting, and legitimately unresolved
evidence.

The current `PROPOSE` also injects ledger uncertainty, exhausted-point redirection,
repair, and deduction instructions into the first call even though both ledger
and feedback are empty. This is a plausible source of unnecessary complexity;
the effect must be tested.

## One compact English candidate

Use the same `STEP_SCHEMA`, constrained JSON generation, and proposal token cap.
Interpolate `{question}` using `task_question(question)`. Generate
`{answer_domain}` solely from the public question/choices, never from a reference
label. For MCQ, an example domain is `one of A, B, C`; for a binary non-MCQ use
`Yes or No`; otherwise use `a short word or number`.

```text
Inspect the image and propose exactly one visual evidence step for this question.
Return one JSON object using the supplied schema:
- kind: "visual".
- claim: one short sentence stating a relevant visible fact, not an option label
  or a full reasoning chain.
- evidence_question: a short neutral question about what should be inspected,
  without suggesting its answer.
- final_answer: if this observation is sufficient to answer the question, use
  {answer_domain}; otherwise use JSON null, not the string "null".
The observation and any proposed answer are tentative and must pass the audit.

Question: {question}
```

This borrows v4 CoT's short, relevant evidence requirement and explicitly scoped
answer formatting, while producing only one structured step. Do not insert
`Evidence:` prose headers or a separate `FINAL ANSWER:` paragraph: this path
already uses JSON fields and does not feed FaithAct's paragraph parser.

## Where it would apply

Select this prompt only when `not ledger and not feedback`. All later calls,
including a parse-error retry, use the existing full proposal/repair instructions.
Keep `STEP_SCHEMA` unchanged; the initial instruction selects `visual` from its
existing enum. The generic deduction option remains available at later steps.

The change would affect the initial call of `Online` and the initial parent call
of `OnlineRepair`. It must not replace `OnlineRepair`'s candidate-blind reread,
erase feedback, reset the three-repair budget, or alter later uncertainty handling.

Any `final_answer` in this JSON remains a candidate. The controller must audit
the visual claim and check the answer before it can return it. A null answer
still supplies a relevant claim and neutral evidence question for auditing; once
accepted, subsequent reasoning can continue from that evidence. A failed audit
still triggers correction, at most three repairs, then an explicit uncertain
point and continued reasoning. The prompt introduces no direct-answer bypass
and never requests a full chain before verification.

## Paired comparison

Use one compact candidate, frozen before the comparison. Compare current V13
with its otherwise identical compact-initial V14, separately for `online_repair`
and `online_cf_repair`. A V9-versus-V13 comparison changes the repair machinery
and cannot isolate this prompt effect.

Keep these settings matched and recorded:

| Setting | Value / requirement |
|---|---|
| Model | Same Qwen2.5-VL-7B-Instruct checkpoint and model revision |
| Precision | NF4 for both engineering arms; BF16 is a separate experiment |
| Image budget | `max_pixels=1003520`, same processor min pixels and crop allocation |
| Decoding | Greedy, same chat template and random seed |
| Proposal budget | Existing explicit `max_tokens=256`, same `STEP_SCHEMA` |
| Job generation budget | Same `max_new_tokens=512`; record per-call overrides too |
| Controller | `max_steps=6`, three repairs after the initial audit |
| Gates / tools | Same support/specificity thresholds, masks, boxes, and option checks |
| Data | Same public MCQ packaging, exact IDs/order, and dataset hash |
| Scoring | One frozen committed-answer extractor for both; raw traces retained |

The first 16 cases are already development examples. Use them for an immediate
paired engineering check, not a generalization claim. If the compact arm shows
a useful gain, keep its wording fixed for a larger paired development comparison
and a separate validation slice whose detailed errors have not been used to
revise this prompt. Report any subsequent full-benchmark score with the declared
development history.

The primary comparison is final end-task accuracy and paired gains/losses between
the two arms. Also report initial valid-answer coverage, initial correctness over
all cases and over valid initial answers, initial deduction/parse failures,
initial-to-final gains/losses on comparable cases only, uncertain/verified counts,
mask/readback/repair counts, latency, and model calls. Show the outcomes of null
or invalid initial answers as a separate group; never count them as initially
wrong to inflate corrections.

An improved initial option score alone is insufficient if the final score or
evidence quality worsens. Conversely, genuine nulls should remain allowed when
the image does not supply the needed evidence. The intended improvement is a
clearer first visual proposal, with the same mandatory verification boundary.
