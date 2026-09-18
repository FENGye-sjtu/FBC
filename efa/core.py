"""Paper controller. Scores must come from a backend, never teacher-authored numbers."""
from dataclasses import dataclass, field, replace
from typing import Protocol
import math


@dataclass(frozen=True)
class Claim:
    subject: str
    predicate: str
    value: str
    family: str = "attribute"
    kind: str = "visual"
    parents: tuple[str, ...] = ()

    @property
    def query(self):
        # The value is deliberately not an input to acquisition.
        return {"subject": self.subject, "predicate": self.predicate}


@dataclass(frozen=True)
class Scores:
    binding: float
    support: float
    deletion: float | None
    specificity: float | None
    sequence_logp: float
    valid: bool = True


@dataclass(frozen=True)
class Policy:
    binding: float
    support: float
    deletion: float = 0.0
    specificity: float = 0.0
    use_deletion: bool = True
    use_specificity: bool = True
    max_rounds: int = 3
    beam_answers: int = 3

    def __post_init__(self):
        if not 0 <= self.max_rounds <= 3 or not 1 <= self.beam_answers <= 3:
            raise ValueError("Paper policy permits at most three rounds and three proposals")

    def failures(self, score: Scores):
        failures = []
        if not score.valid:
            failures.append("invalid")
        for name, value, threshold, enabled in (
            ("binding", score.binding, self.binding, True),
            ("support", score.support, self.support, True),
            ("deletion", score.deletion, self.deletion, self.use_deletion),
            ("specificity", score.specificity, self.specificity, self.use_specificity),
        ):
            # None represents structurally unavailable views; availability must be
            # derived from source geometry, not from the observed claim score.
            if enabled and (value is not None or name in ("binding", "support")):
                if value is None or not math.isfinite(value) or value < threshold:
                    failures.append(name)
        if not math.isfinite(score.sequence_logp):
            failures.append("invalid_logp")
        return failures


@dataclass
class Budget:
    max_steps: int = 16
    max_atoms_per_step: int = 3
    max_review_rounds: int = 12
    steps: int = 0
    rounds: int = 0
    audits: int = 0

    def take_step(self):
        if self.steps >= self.max_steps:
            raise RuntimeError("global_step_budget")
        self.steps += 1


@dataclass
class Resolution:
    claim: Claim
    local_pass: bool
    reason: str
    rounds: int
    events: list[dict] = field(default_factory=list)


class Backend(Protocol):
    def route(self, query: dict): ...
    def score(self, claim: Claim, source) -> Scores: ...
    def refine(self, query: dict, source, failures: list[str], round_id: int): ...
    def propose(self, query: dict, source, k: int) -> list[str]: ...
    def final_context(self, query: dict, source): ...
    def rescore(self, query: dict, source, values: list[str]) -> dict[str, float]: ...


def resolve(claim: Claim, backend: Backend, policy: Policy, budget: Budget) -> Resolution:
    if claim.kind not in ("visual", "premise") or not all(isinstance(s, str) and s.strip() for s in (claim.subject, claim.predicate, claim.value)):
        return Resolution(claim, False, "invalid_claim", 0)
    if hasattr(backend, "begin"):
        backend.begin(claim)
    def provenance(source):
        return backend.describe_source(claim.query, source) if hasattr(backend, "describe_source") else None
    source = backend.route(claim.query)
    score = backend.score(claim, source)
    budget.audits += 1
    failed = policy.failures(score)
    events = [{"round": 0, "value": claim.value, "scores": vars(score), "failures": failed, "source": provenance(source)}]
    if not failed:
        return Resolution(claim, True, "initial_pass", 0, events)
    observed = [claim.value]
    rounds = 0
    for round_id in range(1, policy.max_rounds + 1):
        if budget.rounds >= budget.max_review_rounds:
            break
        budget.rounds += 1
        rounds += 1
        source = backend.refine(claim.query, source, failed, round_id)
        proposed = list(dict.fromkeys(backend.propose(claim.query, source, policy.beam_answers)))
        proposed = [v for v in proposed if isinstance(v, str) and v.strip()][:policy.beam_answers]
        passing = []
        next_failures = []
        for value in proposed:
            if value not in observed:
                observed.append(value)
            candidate = replace(claim, value=value)
            score = backend.score(candidate, source)
            budget.audits += 1
            failures = policy.failures(score)
            events.append({"round": round_id, "value": value, "scores": vars(score), "failures": failures, "source": provenance(source)})
            next_failures.extend(failures)
            if not failures:
                passing.append((value, score.sequence_logp))
        if passing:
            value = sorted(passing, key=lambda x: (-x[1], x[0] != claim.value, x[0]))[0][0]
            return Resolution(replace(claim, value=value), True, "repair_pass", rounds, events)
        failed = list(dict.fromkeys(next_failures)) or ["no_valid_proposal"]
    final = backend.final_context(claim.query, source)
    logps = backend.rescore(claim.query, final, observed)
    valid = [v for v in observed if v in logps and math.isfinite(logps[v])]
    value = sorted(valid, key=lambda v: (-logps[v], v != claim.value, v))[0] if valid else claim.value
    events.append({"action": "common_context_rescore", "logps": logps, "local_pass": False, "source": provenance(final)})
    reason = "review_exhausted" if rounds == policy.max_rounds else "global_review_budget"
    if not valid:
        reason = "rescore_unavailable"
    return Resolution(replace(claim, value=value), False, reason, rounds, events)


@dataclass(frozen=True)
class Node:
    key: str
    claim: Claim
    local_pass: bool
    effective_verified: bool
    parents: tuple[str, ...]
    source_version: str


class Ledger:
    def __init__(self):
        self.live: dict[str, Node] = {}
        self.history: list[Node] = []
        self.versions: dict[str, int] = {}

    def append(self, name, claim, local_pass, exposed_parents, source_version):
        parents = tuple(dict.fromkeys(exposed_parents))
        if any(p not in self.live for p in parents):
            raise ValueError("A parent must be a live version")
        version = self.versions.get(name, 0) + 1
        self.versions[name] = version
        key = f"{name}@{version}"
        effective = bool(local_pass and all(self.live[p].effective_verified for p in parents))
        node = Node(key, claim, local_pass, effective, parents, source_version)
        self.live[key] = node
        self.history.append(node)
        return node

    def revoke(self, key):
        if key not in self.live:
            raise KeyError(key)
        removed = {key}
        while True:
            expanded = removed | {k for k, n in self.live.items() if set(n.parents) & removed}
            if expanded == removed:
                break
            removed = expanded
        for item in removed:
            del self.live[item]
        return removed
