"""Logical eviction invalidates deduplication, but keeps physical hazards."""
import torch
from mgo_v2.pinned_h2d import PriorityH2DScheduler

def main():
 torch.set_num_threads(2);cache=torch.empty((2,9437184//2),device='cuda',dtype=torch.bfloat16);one=torch.ones(cache.shape[1],dtype=cache.dtype);two=one*2
 q=PriorityH2DScheduler(cache)
 first=q.enqueue_demand(0,7,[one]);q.synchronize();q.invalidate(0,7)
 assert not q.ready(0)
 second=q.enqueue_prefetch(0,7,[two]);assert first is not second;q.promote(0,7);q.wait_slots([0],host=True)
 assert bool((cache[0]==2).all()) and q.metrics['copies']==2
 q.enqueue_prefetch(1,8,[one]);q.synchronize();q.discard(1,8)
 old=q.tickets[1];new=q.enqueue_prefetch(1,8,[two]);assert old is not new;q.synchronize()
 assert bool((cache[1]==2).all()) and q.metrics['copies']==4
 q.close();print('PASS logically retired same-key transfers are copied again')
if __name__=='__main__':main()
