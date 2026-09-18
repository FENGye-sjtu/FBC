import unittest
from efa.core import Budget, Claim, Ledger, Policy, Scores, resolve
from efa.regions import Box, complement


class Fake:
    def __init__(self, pass_values=()):
        self.pass_values = set(pass_values)
        self.proposal_calls = 0
        self.rescore_calls = 0
        self.queries = []

    def route(self, query):
        self.queries.append(query)
        return 0

    def score(self, claim, source):
        return Scores(0, 0 if claim.value in self.pass_values else -10, None, None, -1)

    def refine(self, query, source, failures, round_id):
        return round_id

    def propose(self, query, source, k):
        self.proposal_calls += 1
        return ["blue", "black", "red"][:k]

    def final_context(self, query, source):
        return "common"

    def rescore(self, query, source, values):
        assert source == "common"
        self.rescore_calls += 1
        return dict.fromkeys(values, -1)


class ContractTests(unittest.TestCase):
    def test_pass_never_generates_alternatives(self):
        b = Fake(["red"])
        r = resolve(Claim("left bag", "color", "red"), b, Policy(-1,-1), Budget())
        self.assertTrue(r.local_pass)
        self.assertEqual(b.proposal_calls, 0)
        self.assertEqual(b.queries, [{"subject":"left bag", "predicate":"color"}])

    def test_repair_and_bounded_fallback(self):
        c = Claim("left bag", "color", "red")
        r = resolve(c, Fake(["blue"]), Policy(-1,-1), Budget())
        self.assertEqual((r.claim.value,r.local_pass,r.rounds), ("blue",True,1))
        b, budget = Fake(), Budget()
        r = resolve(c,b,Policy(-1,-1),budget)
        self.assertEqual((b.proposal_calls,budget.audits,b.rescore_calls), (3,10,1))
        self.assertFalse(r.local_pass)
        self.assertEqual(r.claim.value,"red")  # common-context tie prefers incumbent

    def test_global_budget_does_not_reset(self):
        budget = Budget(max_review_rounds=1)
        for _ in range(2):
            r = resolve(Claim("bag","color","red"),Fake(),Policy(-1,-1),budget)
            self.assertFalse(r.local_pass)
        self.assertEqual(budget.rounds,1)

    def test_taint_and_rollback(self):
        ledger = Ledger()
        c = Claim("bag","color","red")
        root = ledger.append("a",c,False,[],"v1")
        child = ledger.append("b",c,True,[root.key],"v1")
        independent = ledger.append("c",c,True,[],"v1")
        self.assertFalse(child.effective_verified)
        self.assertEqual(ledger.revoke(root.key), {root.key,child.key})
        self.assertIn(independent.key,ledger.live)
        corrected = ledger.append("a",c,True,[],"v2")
        self.assertEqual(corrected.key,"a@2")

    def test_deletion_has_no_target_pixels(self):
        targets = [Box(10,10,50,50),Box(40,40,80,80)]
        pieces = complement(100,100,targets)
        self.assertTrue(all(not p.overlaps(t) for p in pieces for t in targets))
        self.assertEqual(sum((p.x2-p.x1)*(p.y2-p.y1) for p in pieces),6900)

    def test_nonfinite_never_passes(self):
        p=Policy(-1,-1)
        self.assertTrue(p.failures(Scores(0,float('nan'),None,None,-1)))


if __name__ == '__main__':
    unittest.main()
