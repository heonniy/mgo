"""CPU check of diagnostic traversal, arithmetic and complete hook restoration."""
from types import SimpleNamespace
import torch
from refactor_host_diagnostics import HostDiagnostics
from refactor_expert_diagnostics import install

def main():
 torch.set_num_threads(1)
 class Scheduler:
  def ready(self,slot):return slot==1
  def _submitted(self,t):pass
  def wait_for_slot(self,slot):self._submitted(slot)
  def record_slot_use(self,slot):self.used.append(slot)
  used=[]
 q=Scheduler();cache=torch.zeros((2,4718592),dtype=torch.bfloat16);cache[0,0]=1;cache[1,0]=2
 rt=SimpleNamespace(index=48,args=SimpleNamespace(streaming=True,fused=True,ready_first=True,phase='MEASURE'),h2d=q,cache=cache,keys=[0,1],ready_metrics={'waits':0,'ready_before_first_wait':0},kernel=lambda x,g,u,d:x+g[0])
 fallback=lambda *args:'fallback';rt.compute=fallback
 rows0=torch.tensor([0,2]);rows1=torch.tensor([1]);cols0=torch.tensor([0,0]);cols1=torch.tensor([1])
 groups=[(0,rows0,cols0,0),(1,rows1,cols1,1)]
 x=torch.ones((3,2048),dtype=torch.bfloat16);rw=torch.ones((3,2),dtype=torch.bfloat16)*.5
 expected=[(x[rows0]+cache[0,:2048])*rw[rows0,cols0,None],(x[rows1]+cache[1,:2048])*rw[rows1,cols1,None]]
 original_record=torch.cuda.Event.record;original_ready=q.ready
 diag=HostDiagnostics(lambda:rt.index);uninstall=install(rt,diag);diag.start()
 try:actual=rt.compute(('current',x,None,rw),{'groups':groups},0)
 finally:detail=diag.finish();uninstall()
 assert all(torch.equal(a,b) for a,b in zip(actual,expected)) and q.used==[1,0]
 assert rt.compute is fallback and q.ready==original_ready and torch.cuda.Event.record is original_record
 names={r['phase'] for r in detail['phases']}
 assert {'expert.kernel_call','expert.input_and_weight_views','expert.routing_weight_lookup','scheduler._submitted'}<=names
 print('PASS traversal order, result positions/BF16 arithmetic, nested labels and hook restoration')
if __name__=='__main__':main()
