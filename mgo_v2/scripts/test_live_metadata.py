"""R4 native small-batch Gate history against independent global CPU history."""
import os,json
import numpy as np
import torch
import torch.distributed as dist
from mgo_v2.live_metadata import LiveMetadata
from mgo_v2.eviction import GateHistory

def main():
 rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(1)
 dist.init_process_group('nccl',device_id=torch.device('cuda',rank));assert dist.get_world_size()==4
 checks=0
 for batch in [1,16,64,128,256]:
  actual=GateHistory(48,128,128);reference=GateHistory(48,128,128);metadata=LiveMetadata(batch,actual)
  for generation in range(2):
   rng=np.random.default_rng(123+generation)
   seed=rng.random((321,128),dtype=np.float32);actual.update(0,seed);reference.update(0,seed)
   for step in range(1,5):
    all_probs=rng.random((4*batch,128),dtype=np.float32);all_ids=np.argsort(all_probs,axis=1)[:,-8:].copy()
    got=metadata.collect(step*48,torch.tensor(all_ids[rank*batch:(rank+1)*batch],device='cuda'),torch.tensor(all_probs[rank*batch:(rank+1)*batch],device='cuda'))
    reference.update(0,all_probs);expected=(reference.sums[0]/len(reference.rows[0])).astype(np.float32)
    np.testing.assert_array_equal(got.gate_scores,expected);np.testing.assert_array_equal(got.routes.selected_experts,all_ids)
    assert np.array_equal(got.histogram.sum(axis=1),np.full(4,batch*8));checks+=1
 print(json.dumps(dict(rank=rank,status='PASS',checks=checks)),flush=True);dist.destroy_process_group()
if __name__=='__main__':main()
