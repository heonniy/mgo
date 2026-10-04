"""Native Gate W128 and fixed-record NCCL parity at B128 and B256."""
from env_offload_worker import *
from mgo_v2.compact_metadata import CompactMetadata

def main(a):
 rank=int(os.environ['RANK']);torch.set_num_threads(2);torch.cuda.set_device(0)
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'));rows=[]
 for batch in (128,256):
  torch.manual_seed(rank+batch);probs=torch.softmax(torch.randn(batch,128,device='cuda'),-1)
  weights,selected=torch.topk(probs,8,-1);weights=(weights/weights.sum(-1,keepdim=True)).to(torch.bfloat16)
  compact=CompactMetadata(batch);g=compact.collect(48,selected,probs)
  reference=gather_global_routes(0,selected,weights,probs)
  history=GateHistory(48,128,128);history.update(0,reference.routes.full_router_probs)
  gate=np.array([history.score(0,e) for e in range(128)],np.float32)
  assert np.array_equal(g.gate_scores,gate),'Gate W128 changed'
  assert np.array_equal(g.routes.selected_experts,reference.routes.selected_experts)
  assert np.array_equal(g.routes.origin_ranks,reference.routes.origin_ranks)
  assert compact.calls==1
  row=dict(batch=batch,metadata_collectives=1,bytes_per_rank=compact.record_bytes,native_Gate_W128_exact=True,global_ID_origin_exact=True)
  if a.benchmark:
   samples={'compact':[],'legacy':[]}
   for iteration in range(60):
    for name in (['compact','legacy'] if iteration%2==0 else ['legacy','compact']):
     dist.barrier();torch.cuda.synchronize();start=time.perf_counter()
     if name=='compact':compact.collect(48,selected,probs)
     else:gather_global_routes(0,selected,weights,probs)
     torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
     if iteration>=10:samples[name].append(elapsed)
   import statistics
   row.update(samples_ms=samples,median_ms={k:statistics.median(v) for k,v in samples.items()},timing='synchronized boundaries outside each sample; alternating order; 10 warmup + 50 samples; no model compute in flight')
  rows.append(row)
 write(a.output/f'rank{rank}.json',dict(status='PASS',rank=rank,rows=rows));dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--benchmark',action='store_true');main(p.parse_args())
