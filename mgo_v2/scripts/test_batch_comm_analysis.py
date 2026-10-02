import unittest
from batch_comm_analysis import predictors
from summarize_batch_comm import corr

class GeometryTests(unittest.TestCase):
    def test_asymmetric_traffic_and_self_exclusion(self):
        event=dict(dispatch_send=[[1,2,0,0],[0,0,3,0],[0,0,0,0],[0,0,0,0]],
                   combine_send=[[0,0,0,0],[5,0,0,0],[0,7,2,0],[0,0,0,0]])
        self.assertEqual(predictors(event),dict(total_peer_bytes=17*4096,self_bytes=3*4096,
                         active_remote_ordered_pairs=4,max_rank_fanout=2,max_rank_remote_bytes=8*4096))
        self.assertEqual(predictors(event,'dispatch')['max_rank_remote_bytes'],3*4096)
        self.assertEqual(predictors(event,'combine')['max_rank_remote_bytes'],7*4096)
    def test_union_does_not_double_count_same_edge(self):
        e=dict(dispatch_send=[[0,1,0,0],[0]*4,[0]*4,[0]*4],combine_send=[[0,3,0,0],[0]*4,[0]*4,[0]*4])
        self.assertEqual(predictors(e)['active_remote_ordered_pairs'],1)
        self.assertEqual(predictors(e)['total_peer_bytes'],4*4096)
    def test_constant_predictor_is_unidentifiable(self):
        self.assertIsNone(corr([3,3,3],[1,2,3]))
        self.assertAlmostEqual(corr([1,2,3],[6,4,2]),-1)
if __name__=='__main__':unittest.main()
