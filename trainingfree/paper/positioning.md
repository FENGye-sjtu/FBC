# Research claim and empirical priorities

The user's invariant is online evidence verification with local repair, not a
particular prompt or score threshold. The backbone stays frozen. Three failed
repair rounds lead to explicitly uncertain continuation, never automatic rejection
of the whole problem or silent acceptance of a low-confidence fact.

## What must be new

FaithAct's **paper** already describes immediate step validation and action-guided
refinement (Sections 4.2–4.3). Its **released Qwen script** consumes a precomputed
whole CoT, extracts objects, then regenerates an answer. These are different
levels of description. We must not write that all previous work only verifies
after completion, or use the released script's limitation to erase its paper's
claimed online design.

Our candidate contribution is therefore the combination of:

1. Checking the actual attribute/relation assertion, not only object existence.
2. Lightweight localized counterfactual specificity using the frozen reasoner,
   without a separate detector/classifier verification stack.
3. A concrete step-boundary controller that replaces rejected drafts, reacquires
   evidence and carries unresolved status forward after bounded repair.

These are implementation choices and hypotheses until the experiments support
them. Current occlusion specificity is a proxy, not causal identification or a
proof that every accepted statement is true.

## Accuracy-first next experiments

- Complete matched CoT / released-FaithAct port / online / online-CF development
  runs. Fix concrete faults revealed in traces, without choosing thresholds to
  fit test answers. Small development examples do not establish superiority.
- Finish BF16 CoT and FaithAct on 765 original RealWorldQA examples and explicitly
  compare against the paper's 70.1% and 74.5%, with unresolved protocol differences.
- Freeze a method configuration and evaluate the same complete benchmark plus
  MMStar. Label examples previously inspected during development; report both
  whole-set and uninspected-subset outcomes, not an invented held-out benchmark.
- Report wrong-to-correct and correct-to-wrong answer changes, repair frequency,
  uncertain continuation and cost. Inspect actual CF executions; final accuracy
  alone can conceal a verifier that never runs or always falls back.
- Once useful, add dependency-targeted rollback. It is not implemented or claimed
  in the current submission draft and must not delay the baseline replication.

There is no training run in this track. The old repository's SFT work is separate.
