import unittest
import numpy as np
from replica_pareto_cpu import ReplicaReplay,traffic,ROW_BYTES,EXPERT_BYTES
from rank_local_reuse import marginal_pairs,next_distance,burst,future_stats,victim_penalty,adjusted_score

class ReuseTests(unittest.TestCase):
    def test_exact_marginal_dispatch_and_combine(self):
        origins=np.array([0,0,1]);selected=np.array([[1,2],[1,3],[1,2]]);dests=np.array([[1,1],[1,0],[1,1]])
        records=marginal_pairs(origins,selected,dests)
        # e1: two remote rows, but only the second removes the last remote route.
        self.assertEqual(records[1,0]['marginal_bytes'],3*ROW_BYTES)
        self.assertEqual(records[2,0]['marginal_bytes'],ROW_BYTES)
        self.assertEqual(records[1,1]['marginal_bytes'],0)
        for (expert,rank),r in records.items():
            trial=dests.copy();mask=(selected==expert)&(origins[:,None]==rank);trial[mask]=rank
            self.assertEqual(traffic(origins,dests,2)['peer_bytes']-traffic(origins,trial,2)['peer_bytes'],r['marginal_bytes'])
    def test_future_excludes_current_and_reuse_distance(self):
        line=dict(remote_routes=[0,2,0,1,3,0,0,0,1],demand_routes=[0,2,1,1,3,0,0,0,1],marginal_bytes=[0,8192,0,4096,16384,0,0,0,4096])
        self.assertEqual(next_distance(line['remote_routes'],1),2)
        self.assertEqual(burst(line['remote_routes'],3),2)
        self.assertEqual(future_stats(line,1,1)['future_peer_bytes_saved'],0)
        self.assertEqual(future_stats(line,1,2)['future_peer_bytes_saved'],4096)
        self.assertEqual(future_stats(line,1,'remaining')['future_peer_bytes_saved'],24576)
        self.assertIsNone(next_distance(line['remote_routes'],8))
        self.assertEqual(future_stats(line,8,'remaining')['reuse_steps'],0)
    def test_victim_unique_copy_and_horizon(self):
        p=ReplicaReplay([2,2],1);p.place(0,(0,1),(0,None));choice=(0,(0,1))
        self.assertEqual(victim_penalty(p,choice,49,1,{(0,1):[96]}),1)
        self.assertEqual(victim_penalty(p,choice,49,1,{(0,1):[98]}),0)
        p.place(1,(0,1),(0,None))
        self.assertEqual(victim_penalty(p,choice,49,1,{(0,1):[96]}),0)
        self.assertEqual(victim_penalty(p,(1,None),49,1,{(0,1):[96]}),0)
        self.assertEqual(adjusted_score(131072,1,65536),0)
        self.assertEqual(adjusted_score(139264,1,65536),69632)
        self.assertEqual(adjusted_score(EXPERT_BYTES+4096,1,0,'subtract_expert_bytes'),4096)
    def test_deterministic_future_ties_keep_actual_traffic(self):
        a=ReplicaReplay([4,4],1);b=ReplicaReplay([4,4],1)
        score=lambda *args: 1024
        ra,da,_,oa=a.event(0,[0,1],[[0,1],[0,1]],candidate_score=score)
        rb,db,_,ob=b.event(0,[0,1],[[0,1],[0,1]],candidate_score=score)
        self.assertEqual([(o[1],o[2]) for o in oa if o[0]=='replica'],[(0,1),(1,1)])
        self.assertEqual(ra['greedy_peer_bytes_saved'],3*ROW_BYTES)
        self.assertEqual(ra,rb);self.assertEqual(oa,ob);np.testing.assert_array_equal(da,db)

if __name__=='__main__':unittest.main()
