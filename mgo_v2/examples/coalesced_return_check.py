"""Two-payload-A2A candidate against canonical high-precision contribution sum."""
import os,json,argparse
from mgo_v2.bootstrap import pin_rank_before_cuda_import
BOOT=pin_rank_before_cuda_import()
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from env_offload_layout import plan_layout
from env_offload_tensors import pack_layouts
from mgo_v2.fused_transport import FusedTokenRankTransport
from mgo_v2.coalesced_return import combine_rank_partials

def run(a):
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.use_deterministic_algorithms(True);dist.init_process_group('nccl',device_id=torch.device('cuda:0'));rank=dist.get_rank();torch.manual_seed(100+rank);rows=[]
 for batch in (128,256):
  for case in ('random','uneven','zero-peers'):
   rng=np.random.default_rng(42);counts=[batch+(r%3 if case=='uneven' else 0) for r in range(8)];origins=np.repeat(np.arange(8),counts)
   selected=np.argsort(rng.random((sum(counts),128)),axis=1)[:,:8];owner=np.zeros(128,np.int64) if case=='zero-peers' else rng.integers(0,8,128)
   e=plan_layout(selected,np.full(sum(counts),8),owner[selected],origins,counts,rank);e.update(targets=np.arange(128),selected=selected[sum(counts[:rank]):sum(counts[:rank+1])]);e['groups']=[(*g,0) for g in e['groups']];e=pack_layouts([e])[0]
   hidden=torch.randn((counts[rank],2048),device='cuda',dtype=torch.bfloat16);dense=torch.rand((counts[rank],128),device='cuda',dtype=torch.bfloat16)
   reference_transport=FusedTokenRankTransport();rh,_=reference_transport.exchange(hidden[e['send_idx']],e['send_counts'],e['recv_counts']);sw=dense[e['send_idx'][:,None],e['send_eids'].clamp_min(0)]*(e['send_eids']>=0);rw,_=reference_transport.exchange(sw,e['send_counts'],e['recv_counts'])
   parts=[(rh[r]*((expert+1)/128))*rw[r,c,None] for expert,r,c,_ in e['groups']]
   values=torch.cat(parts) if parts else hidden.new_empty((0,2048));returned,_=reference_transport.exchange(values[e['return_order']],e['return_counts'],e['return_recv_counts']);canonical=torch.zeros_like(hidden,dtype=torch.float64);legacy=torch.zeros_like(hidden)
   for idx,pos in e['combine']:
    canonical.index_add_(0,idx,returned[pos].double());legacy.index_add_(0,idx,returned[pos])
   canonical=canonical.bfloat16()
   for dtype in (torch.bfloat16,torch.float32,torch.float64):
    transport=FusedTokenRankTransport();fh,fw,fi=transport.forward(hidden,dense,e).finish();assert torch.equal(fh,rh) and torch.equal(fw,rw)
    for expert,r,c,_ in e['groups']:assert bool((fi[r,c]==expert).all())
    newparts=[(fh[r]*((expert+1)/128))*fw[r,c,None] for expert,r,c,_ in e['groups']]
    output=combine_rank_partials(transport,hidden,newparts,e,dtype);assert transport.calls==2
    delta=(output.float()-canonical.float()).abs();relative_l2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(canonical.float()).clamp_min(1e-20));diff=int((output!=canonical).count_nonzero())
    if dtype==torch.float64:assert diff==0
    rows.append(dict(batch=batch,case=case,accumulation=str(dtype),payload_collectives=transport.calls,return_rows=sum(e['recv_counts']),diff_vs_canonical=diff,relative_L2_vs_canonical=relative_l2,max_abs_vs_canonical=float(delta.max()),diff_vs_legacy=int((output!=legacy).count_nonzero()),forward_bytes=transport.forward_bytes,return_bytes=transport.return_bytes))
 a.output.mkdir(parents=True,exist_ok=True);(a.output/f'rank{rank}.json').write_text(json.dumps(dict(status='PASS',rank=rank,rows=rows,scope='contribution identity and two-A2A physical transport; model token accuracy not tested here'),indent=2)+'\n');dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args())
