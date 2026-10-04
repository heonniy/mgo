from mgo_v2.active_peer import active_peer_set,split_offsets
def test_active_peers_ignore_self_and_zero_pairs():assert active_peer_set([3,0,5,0],[3,2,0,0],0)==(1,2)
def test_split_offsets():assert split_offsets([2,0,3,1])==[0,2,2,5,6]
