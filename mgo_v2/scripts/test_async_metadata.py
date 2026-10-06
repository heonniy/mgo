"""Four-rank metadata parity with rapidly reused pinned input buffers."""
import json,os
import numpy as np
import torch
import torch.distributed as dist
from mgo_v2.compact_metadata import CompactMetadata

def main():
 rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(1)
 dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
 assert dist.get_world_size()==4
 checks=0
 for batch in (16,64,128,256):
  old=CompactMetadata(batch);new=CompactMetadata(batch,async_inputs=True)
  for event in range(16):
   torch.manual_seed(1700+event+rank*100)
   ids=torch.randint(0,128,(batch,8),device='cuda')
   probs=torch.rand((batch,128),device='cuda',dtype=torch.bfloat16)
   gate=np.arange(128,dtype=np.float32)*(event+1)/127 if event%2 or batch<128 else None
   # Alternate call order; each implementation must preserve metadata bytes,
   # rank-origin order and Gate tail across buffer reuse and fresh inputs.
   if event%2:a=new.collect(event,ids,probs,gate);b=old.collect(event,ids,probs,gate)
   else:b=old.collect(event,ids,probs,gate);a=new.collect(event,ids,probs,gate)
   for field in ('counts','local_offset'):assert getattr(a,field)==getattr(b,field)
   for field in ('gate_scores','histogram'):assert np.array_equal(getattr(a,field),getattr(b,field))
   for field in ('selected_experts','routing_weights','origin_ranks'):assert np.array_equal(getattr(a.routes,field),getattr(b.routes,field))
   assert torch.equal(old.host,new.host)
   if gate is not None:assert np.array_equal(a.gate_scores,gate)
   expected=torch.empty((4*batch,8),dtype=ids.dtype,device='cuda');dist.all_gather_into_tensor(expected,ids)
   assert np.array_equal(a.routes.selected_experts,expected.cpu().numpy())
   checks+=1
  assert old.calls==new.calls==16
  if batch<128:
   for item in (old,new):
    try:item.collect(999,ids,probs)
    except ValueError as exc:assert 'requires frozen gate' in str(exc)
    else:raise AssertionError('small-batch live Gate history must not be silently truncated')
 dist.barrier()
 print(json.dumps(dict(status='PASS',rank=rank,checks=checks,full_record_byte_parity=True,batches=[16,64,128,256],frozen_and_dynamic_gate=True,pinned_reuse=True)),flush=True)
 dist.destroy_process_group()
if __name__=='__main__':main()
