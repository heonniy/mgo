"""Pack immutable workload routes and exact live-controller reference schedules."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from ttft_common import *
from ttft_screen import order_for
from env_offload_policy import Policy
import numpy as np
import concurrent.futures,multiprocessing

def workload(batch,wseed):
 dst=ROOT/'workloads'/f'B{batch}_w{wseed}'
 if (dst/'receipt.json').exists():return dst
 dst.mkdir(parents=True,exist_ok=True);order=order_for(batch,wseed);n=4*batch*512
 for name,dtype in [('selected','uint8'),('weights','float32')]:
  src=np.load(ROOT/'pool'/f'{name}.npy',mmap_mode='r');out=np.lib.format.open_memmap(dst/f'{name}.npy',mode='w+',shape=(48*n,8),dtype=dtype)
  for layer in range(48):out[layer*n:(layer+1)*n]=src[order,layer].reshape(n,8)
  out.flush()
 g=np.load(ROOT/'pool/gates.npy',mmap_mode='r');np.save(dst/'gates.npy',g[order[-1]])
 np.save(dst/'offsets.npy',np.arange(49,dtype=np.int64)*n);np.save(dst/'prefill_origins.npy',np.repeat(np.arange(4,dtype=np.int8),batch*512));np.save(dst/'decode_origins.npy',np.repeat(np.arange(4,dtype=np.int8),batch))
 requests=json.loads((ROOT/'requests.json').read_text())['requests'];write(dst/'requests.json',dict(ranks=[[requests[int(q)] for q in rank] for rank in order.reshape(4,batch)],context=512))
 write(dst/'receipt.json',dict(status='PASS',batch=batch,workload_seed=wseed,context=512,request_ids=order.tolist(),files={f.name:sha(f) for f in dst.glob('*.npy')}))
 return dst

def prepare(spec):
 cache,batch,wseed,pseed=spec;dst=ROOT/'inputs'/f'C{cache}_B{batch}_w{wseed}_p{pseed}'
 if (dst/'receipt.json').exists():assert json.loads((dst/'receipt.json').read_text())['status']=='PASS';return str(dst)
 src=ROOT/'workloads'/f'B{batch}_w{wseed}';assert json.loads((src/'receipt.json').read_text())['status']=='PASS';dst.mkdir(parents=True,exist_ok=True)
 for f in [*src.glob('*.npy'),src/'requests.json']:
  if not (dst/f.name).exists():(dst/f.name).symlink_to(f)
 arrays={k:np.load(dst/f'{k}.npy',mmap_mode='r') for k in ('selected','weights','offsets','prefill_origins','gates')}
 total=48*128*cache//100;caps=[total//4+(r<total%4) for r in range(4)];proofs={}
 for policy,kind in [('BR',0),('CA',1),('OLD_CA',3),('LA',4)]:
  c=Policy(caps,np.zeros((48,128,128),np.float32),False,kind,pseed);fetches=[];copies=np.zeros(4,np.int64);layer_counts=[]
  for layer in range(48):
   lo,hi=arrays['offsets'][layer:layer+2];out=c.apply(layer,arrays['selected'][lo:hi],arrays['weights'][lo:hi],arrays['prefill_origins'],arrays['gates'][layer],np.zeros((128,4),np.int32))
   fs=[[int(x) for x in f] for f in out[5]];fetches.append(fs);counts=np.bincount([f[0] for f in fs],minlength=4);copies+=counts;layer_counts.append(counts.tolist());assert counts.max()-counts.min()<=1 and all(f[-1]==0 for f in fs)
  path=dst/f'{policy}_fetches.json';write(path,fetches)
  proof=dict(status='PASS',rank_state_hashes=[__import__('hashlib').sha256(np.ascontiguousarray(c.slots[r,:caps[r]]).tobytes()).hexdigest() for r in range(4)],H2D_bytes=(copies*9437184).tolist(),copies=copies.tolist(),layer_fetch_counts=layer_counts,fetches_sha256=sha(path),substitution=False,replication=False)
  proofs[policy]=proof
 write(dst/'receipt.json',dict(status='PASS',cache=cache,batch=batch,workload_seed=wseed,placement_seed=pseed,context=512,capacities=caps,proofs=proofs,workload_receipt_sha256=sha(src/'receipt.json')));return str(dst)

def main():
 rows=json.loads((PACKET/'TTFT_S0_SHORTLIST.json').read_text())['rows'];specs=sorted({(r['cache'],r['batch'],r['workload_seed'],r['placement_seed']) for r in rows});workloads=sorted({(s[1],s[2]) for s in specs})
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
  list(pool.map(_workload,workloads))
  for path in pool.map(prepare,specs):print('prepared',path,flush=True)
 write(PACKET/'TTFT_INPUTS.json',dict(status='PASS',input_count=len(specs),workload_count=len(workloads),paths=[str(ROOT/'inputs'/f'C{c}_B{b}_w{w}_p{p}') for c,b,w,p in specs]))
def _workload(spec):return str(workload(*spec))
if __name__=='__main__':main()
