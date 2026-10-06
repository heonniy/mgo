"""Prepare static owner proofs against existing selected decode32 traces."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import json,hashlib,shutil
import numpy as np
from strict_headroom_common import *
from strict_static_mod_policy import StaticPolicy

def array_hash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def prepare(cell):
 import psutil
 assert psutil.virtual_memory().available>256*2**30
 world,batch,context=cell;source=ROOT/'physical'/f'S2D32_R{world}_B{batch}_L{context}';result=json.loads((source/'result.json').read_text());assert result['status']=='PASS'
 base=source/'frozen';receipt=json.loads((base/'receipt.json').read_text());assert receipt['decode_steps']==32
 dst=ROOT/'static_inputs'/f'R{world}_B{batch}_L{context}';dst.mkdir(parents=True,exist_ok=False)
 for p in base.iterdir():
  if p.suffix=='.npy' or p.name=='requests.json':(dst/p.name).symlink_to(p.resolve())
 for rank in range(world):shutil.copyfile(source/f'BR_warm_rank{rank}.json',dst/f'reference_BR_warm_rank{rank}.json')
 arrays={k:np.load(dst/(k+'.npy'),mmap_mode='r') for k in ('selected','weights','offsets','prefill_origins','decode_origins','gates')}
 c=StaticPolicy(receipt['capacities'],np.zeros((48,128,128),np.float32),False,8,receipt['placement_seed']);fs=[];counts=[];rank_rows=[]
 for event in range(1584):
  lo,hi=arrays['offsets'][event:event+2];out=c.apply(event,arrays['selected'][lo:hi],arrays['weights'][lo:hi],arrays['prefill_origins' if event<48 else 'decode_origins'],arrays['gates'][event],np.zeros((128,world),np.int32))
  fetch=[[int(v) for v in x] for x in out[5]]
  assert all(rank==(key%128)%world and not replica for rank,key,slot,victim,replica in fetch)
  active=np.flatnonzero(c.owner);assert all(int(c.primary[key])==(key%128)%world and int(c.owner[key])==1<<((key%128)%world) for key in active)
  fs.append(fetch);counts.append(np.bincount([x[0] for x in fetch],minlength=world).tolist());valid=out[4][out[4]>=0];rank_rows.append(np.bincount(valid,minlength=world).tolist())
 fp=dst/'STATIC_MOD_fetches.json';write(fp,fs);copies=np.array(counts).sum(0)
 proof=dict(status='PASS',rank_state_hashes=[array_hash(c.slots[r,:receipt['capacities'][r]]) for r in range(world)],copies=copies.tolist(),H2D_bytes=(copies*9437184).tolist(),layer_fetch_counts=counts,fetches_sha256=sha(fp),owner_rule='expert_id % world',rank_expert_rows=np.sum(rank_rows,axis=0).tolist(),balanced_miss_quota=False)
 receipt.update(proofs={'STATIC_MOD':proof},static_source=str(base),reference_result=str(source/'result.json'),reference_result_sha256=sha(source/'result.json'))
 write(dst/'receipt.json',receipt);return str(dst)

def main():
 import concurrent.futures,multiprocessing
 cells=[(w,b,l) for w in (4,8) for b in (16,64) for l in (256,512)]
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
  paths=list(pool.map(prepare,cells))
 write(PACKET/'STATIC_MOD_INPUTS.json',dict(status='PASS',paths=paths,trace_recapture=False,owner_rule='expert_id % logical world size'))
if __name__=='__main__':main()
