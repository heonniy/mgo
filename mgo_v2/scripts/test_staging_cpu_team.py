"""Full copy correctness and fixed helper placement across worker recreation."""
import os,torch
from mgo_v2.pinned_h2d import PriorityH2DScheduler

def main():
 torch.set_num_threads(2);cpus=sorted(os.sched_getaffinity(0))[1:3]
 n=9*1024*1024//2;source=torch.arange(n,dtype=torch.int32).remainder(127).to(torch.bfloat16)
 cache=torch.empty((2,n),device='cuda',dtype=torch.bfloat16)
 receipts=[]
 for repeat in range(2):
  scheduler=PriorityH2DScheduler(cache,cpu_team=cpus)
  try:
   team=scheduler.cpu_team_receipt
   for slot in range(2):scheduler.enqueue_demand(slot,slot,[source[:n//3],source[n//3:2*n//3],source[2*n//3:]])
   scheduler.wait_slots([0,1],host=True);scheduler.synchronize()
   assert torch.equal(cache[0].cpu(),source) and torch.equal(cache[1].cpu(),source)
   assert sorted(os.sched_getaffinity(team['staging_tid']))==[cpus[0]]
   assert sorted(os.sched_getaffinity(team['helper_tid']))==[cpus[1]]
   assert scheduler.metrics['copies']==2 and scheduler.metrics['bytes']==2*n*2
   receipts.append(team)
  finally:scheduler.close()
 assert receipts[0]['staging_tid']!=receipts[1]['staging_tid']
 print('PASS: BF16 9-MiB copies, fixed main/helper masks, and scheduler recreation')
if __name__=='__main__':main()
