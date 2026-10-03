import unittest
import numpy as np
from hot_expert_cpu import current_pairs
from replica_pareto_cpu import traffic

class HotnessTests(unittest.TestCase):
    def test_exact_counterfactual_and_bound(self):
        o=np.array([0,0,1,2]);s=np.array([[1,2],[1,3],[1,2],[2,3]])
        owners={1:1,2:1,3:2};d=np.array([[owners[int(e)] for e in row] for row in s])
        event=dict(step=1,layer=0,origins=o.tolist(),selected=s.tolist(),destinations=d.tolist(),cache_before=[0,[],[[[0,e],r] for e,r in owners.items()],[]])
        rows=list(current_pairs(event))
        self.assertEqual(sum(r['n'] for r in rows),s.size)
        for r in rows:
            trial=d.copy();trial[(s==r['expert'])&(o[:,None]==r['rank'])]=r['rank']
            a=traffic(o,d,4);b=traffic(o,trial,4)
            self.assertEqual(r['dispatch_bytes_saved'],a['dispatch_bytes']-b['dispatch_bytes'])
            self.assertEqual(r['combine_bytes_saved'],a['combine_bytes']-b['combine_bytes'])
            self.assertLessEqual(r['peer_bytes_saved'],2*r['n']*4096)
        shared=next(r for r in rows if r['expert']==1 and r['rank']==0)
        self.assertEqual(shared['n'],2);self.assertEqual(shared['shared_owner_routes'],1)
        self.assertEqual(shared['peer_bytes_saved'],3*4096)
if __name__=='__main__':unittest.main()
