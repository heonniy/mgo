"""Physical reference/fused parity, including non-associative BF16 negative control."""
import os,json,argparse
from pathlib import Path
from mgo_v2.bootstrap import pin_rank_before_cuda_import
BOOT=pin_rank_before_cuda_import()
import numpy as np
import torch
import torch.distributed as dist
from env_offload_layout import plan_layout
from env_offload_tensors import pack_layouts
from mgo_v2.fused_transport import FusedTokenRankTransport

def run(a):
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.use_deterministic_algorithms(True);dist.init_process_group('nccl',device_id=torch.device('cuda:0'));rank=dist.get_rank();torch.manual_seed(100+rank);rows=[];compressed_mismatches=0
 for batch in (128,256):
  for case in ('random','uneven','zero-peers'):
   rng=np.random.default_rng(42);counts=[batch+(r%3 if case=='uneven' else 0) for r in range(8)];origins=np.repeat(np.arange(8),counts)
   selected=np.argsort(rng.random((sum(counts),128)),axis=1)[:,:8];owner=np.zeros(128,np.int64) if case=='zero-peers' else rng.integers(0,8,128);dest=owner[selected]
   e=plan_layout(selected,np.full(sum(counts),8),dest,origins,counts,rank);e.update(targets=np.arange(128),selected=selected[sum(counts[:rank]):sum(counts[:rank+1])]);e['groups']=[(*g,0) for g in e['groups']];e=pack_layouts([e])[0]
   hidden=torch.randn((counts[rank],2048),device='cuda',dtype=torch.bfloat16);dense=torch.rand((counts[rank],128),device='cuda',dtype=torch.bfloat16)
   ref=FusedTokenRankTransport();rh,_=ref.exchange(hidden[e['send_idx']],e['send_counts'],e['recv_counts']);sw=dense[e['send_idx'][:,None],e['send_eids'].clamp_min(0)]*(e['send_eids']>=0);rw,_=ref.exchange(sw,e['send_counts'],e['recv_counts'])
   parts=[(rh[r]*((expert+1)/128))*rw[r,c,None] for expert,r,c,_ in e['groups']];reference=ref.combine(hidden,parts,e);assert ref.calls==3
   for mode in ('rank-partial','exact'):
    fused=FusedTokenRankTransport(mode);packet=fused.forward(hidden,dense,e);fh,fw,fi=packet.finish();assert torch.equal(fh,rh) and torch.equal(fw,rw)
    for expert,r,c,_ in e['groups']:assert bool((fi[r,c]==expert).all())
    newparts=[(fh[r]*((expert+1)/128))*fw[r,c,None] for expert,r,c,_ in e['groups']]
    output=fused.combine(hidden,newparts,e);assert fused.calls==2
    differing=int((reference!=output).count_nonzero());delta=float((reference.float()-output.float()).abs().max())
    if mode=='exact':assert differing==0
    else:compressed_mismatches+=differing
    rows.append(dict(batch=batch,case=case,mode=mode,differing_elements=differing,max_abs_difference=delta,payload_collectives=fused.calls,forward_bytes=fused.forward_bytes,return_bytes=fused.return_bytes))
 assert compressed_mismatches>0,'negative control must exercise BF16 reassociation'
 a.output.mkdir(parents=True,exist_ok=True);(a.output/f'rank{rank}.json').write_text(json.dumps(dict(status='PASS',rank=rank,rows=rows,rank_partial_gate='FAIL_BITWISE',exact_gate='PASS_BITWISE'),indent=2)+'\n');dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args())
