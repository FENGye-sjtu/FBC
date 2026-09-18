import unittest
from efa.core import Claim,Ledger,Budget,Policy
from efa.infer import revisit
from test_contract import Fake


class RevisitTests(unittest.TestCase):
    def test_correction_revokes_suffix_and_keeps_budget(self):
        ledger=Ledger();claim=Claim('left bag','color','red')
        old=ledger.append('step1.atom0',claim,False,[],'source')
        child=ledger.append('step2.atom0',claim,True,[old.key],'source')
        independent=ledger.append('step3.atom0',claim,True,[],'source')
        budget=Budget(rounds=5)
        backend=Fake(['blue'])
        result=revisit(ledger,old.key,backend,{'default':Policy(-1,-1)},budget)
        self.assertTrue(result['changed'])
        self.assertEqual(set(result['revoked']),{old.key,child.key})
        self.assertIn(independent.key,ledger.live)
        self.assertEqual(result['replacement']['key'],'step1.atom0@2')
        self.assertEqual(result['replacement']['claim']['value'],'blue')
        self.assertEqual(backend.premises,[])
        self.assertEqual(budget.rounds,6)


if __name__=='__main__':
    unittest.main()
