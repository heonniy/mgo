import unittest
import numpy as np

from mgo_v2.fanout_admission import fanout_assignment, load_fanout_assignment


class TestFanoutAdmission(unittest.TestCase):
    def test_fca_coalesces_token_rank_packets_under_balanced_quota(self):
        effective=np.array([[0,1],[0,1],[2,3],[2,3]],dtype=np.int16)
        lengths=np.array([2,2,2,2],dtype=np.int8)
        origins=np.array([0,0,1,1],dtype=np.int8)
        primary=np.full(4,-1,dtype=np.int8)
        misses=np.array([0,1,2,3],dtype=np.int64)
        got=fanout_assignment(effective,lengths,origins,primary,0,4,misses,2)
        self.assertEqual(got.tolist(),[0,0,1,1])
        self.assertEqual(np.bincount(got,minlength=2).tolist(),[2,2])

    def test_la_ca_splits_hot_compute_before_fanout_tiebreak(self):
        effective=np.array([[0],[0],[0],[0],[1],[1],[1],[1],[2],[3]],dtype=np.int16)
        lengths=np.ones(10,dtype=np.int8)
        origins=np.array([0,0,0,0,0,0,0,0,1,1],dtype=np.int8)
        demand=np.zeros((4,2),dtype=np.int64)
        for t in range(10):demand[effective[t,0],origins[t]]+=1
        owner=np.zeros(4,dtype=np.int16)
        primary=np.full(4,-1,dtype=np.int8)
        misses=np.array([0,1,2,3],dtype=np.int64)
        got=load_fanout_assignment(demand,effective,lengths,origins,owner,primary,0,4,misses,2)
        self.assertEqual(np.bincount(got,minlength=2).tolist(),[2,2])
        self.assertNotEqual(got[0],got[1])
        loads=[0,0]
        for i,e in enumerate(misses):loads[int(got[i])]+=int(demand[e].sum())
        self.assertEqual(loads,[5,5])


if __name__ == "__main__":
    unittest.main()
