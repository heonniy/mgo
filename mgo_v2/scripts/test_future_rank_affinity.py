import unittest
import numpy as np
from future_rank_affinity import owner_costs, PlacementReplay
from replica_pareto_cpu import traffic

class PlacementTests(unittest.TestCase):
    def test_exact_shared_dispatch(self):
        origins=np.array([0,1,2,0]);selected=np.array([[1,2],[1,3],[1,2],[2,3]])
        dest=np.array([[2,2],[2,1],[2,2],[2,1]])
        costs=owner_costs(origins,selected,dest,1)
        full=[]
        for r in range(4):
            trial=dest.copy();trial[selected==1]=r
            full.append(traffic(origins,trial,4)['peer_bytes'])
        self.assertEqual(len(set(np.array(full)-costs)),1)

    def test_future_positive_candidates_and_no_migration(self):
        o=np.array([0,1]);s=np.array([[3],[3]])
        r=PlacementReplay([2]*4,'OH1',{(48,3,1):np.array([90000,0,-999999,-999999])},{(0,3):[48,96]})
        _,d,_,_=r.step(48,0,o,s)
        self.assertTrue(np.all(d==1))
        self.assertEqual(r.decisions[0]['F_owner'],0)
        r.step(96,0,o,s)
        self.assertEqual(len(r.decisions),1)
        self.assertTrue(r.finish()[0]['survived_next_demand'])

    def test_tie_and_active_protection(self):
        r=PlacementReplay([1]*4,'O0',{}, {(0,3):[48],(0,4):[49]})
        r.step(48,0,np.array([0,1]),np.array([[3],[3]]))
        self.assertEqual(r.decisions[0]['owner'],0)
        r.step(49,0,np.array([0]),np.array([[4]]))
        self.assertEqual(r.evictions,[1,0,0,0])
        self.assertFalse(r.finish()[0]['survived_next_demand'])

if __name__=='__main__':unittest.main()
