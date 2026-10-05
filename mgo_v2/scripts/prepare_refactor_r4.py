"""Owner R4/decode64 pivot: reuse exact inputs, no GPU or trace recapture."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import json,hashlib,concurrent.futures
from pathlib import Path
import numpy as np
import psutil
from mgo_v2.controller import DecodePrefetchController
from mgo_v2.predictor import TransitionPredictor
BASE=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004')
ROOT=BASE/'R4_H64'
SOURCE=Path('/home/hwlee/mgo-results/la_physical_validation_20261004')
PACKET=Path(__file__).resolve().parents[1]/'experiments/decode_prefetch_runtime_refactoring_20261004'
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def prepare(batch):
 src=SOURCE/f'B{batch}';dst=ROOT/f'inputs_B{batch}_H64';dst.mkdir(parents=True,exist_ok=True)
 if (dst/'input_receipt.json').exists():return
 a={k:np.load(src/f'{k}.npy',mmap_mode='r') for k in ('selected','weights','offsets','prefill_origins','decode_origins','gates','teacher')}
 orgs=[a['prefill_origins'],a['decode_origins']];masks=[o>=4 for o in orgs]
 assert all(np.array_equal(np.unique(o),np.arange(8)) for o in orgs)
 assert all(np.all(o[:-1]<=o[1:]) for o in orgs)
 events=65*48;sizes=[int(masks[0 if e<48 else 1].sum()) for e in range(events)];offsets=np.r_[0,np.cumsum(sizes)]
 for key in ('selected','weights'):
  out=np.lib.format.open_memmap(dst/f'{key}.npy',mode='w+',dtype=a[key].dtype,shape=(int(offsets[-1]),*a[key].shape[1:]))
  for event in range(events):
   lo,hi=a['offsets'][event:event+2];mask=masks[0 if event<48 else 1]
   assert hi-lo==len(mask)
   out[offsets[event]:offsets[event+1]]=a[key][lo:hi][mask]
  out.flush();del out
 np.save(dst/'offsets.npy',offsets)
 for key,o,mask in zip(('prefill_origins','decode_origins'),orgs,masks):np.save(dst/f'{key}.npy',o[mask]-4)
 np.save(dst/'gates.npy',a['gates'][:events]);np.save(dst/'teacher.npy',a['teacher'][4*batch:8*batch,:64])
 requests=json.loads((src/'requests.json').read_text());ranks=requests['ranks'][4:8]
 assert all(len(r)==batch for r in ranks)
 for rank,rows in enumerate(ranks):
  for row in rows:row['source_logical_rank']=row['logical_rank'];row['logical_rank']=rank
 write(dst/'requests.json',dict(ranks=ranks,source_sha256=sha(src/'requests.json'),source_logical_ranks=[4,5,6,7]))
 old=json.loads((src/'receipt.json').read_text())
 write(dst/'input_receipt.json',dict(status='PASS',world=4,batch=batch,horizon=64,physical_gpus=[0,1,4,5],source_logical_ranks=[4,5,6,7],placement_seed=old['winner']['placement_seed'],source_receipt_sha256=sha(src/'receipt.json'),gate_scope='Frozen original Gate W128 scores; source final rank7 retained and local B>=128, so final rank tail is unchanged. No new native-gate computation.',files={p.name:sha(p) for p in dst.glob('*.npy')}))
 print('input prepared',batch,flush=True)
def proof(spec):
 batch,policy,budget=spec;dst=ROOT/f'inputs_B{batch}_H64';target=dst/f'{policy}_P{budget}_proof.json'
 if target.exists():return json.loads(target.read_text())
 assert psutil.virtual_memory().available>256*2**30
 meta=json.loads((dst/'input_receipt.json').read_text());a={k:np.load(dst/f'{k}.npy',mmap_mode='r') for k in ('selected','weights','offsets','prefill_origins','decode_origins','gates')}
 caps=[922,922,921,921];c=DecodePrefetchController(caps,budget,policy,meta['placement_seed'],TransitionPredictor(np.load(BASE/'predictor/transition.npy')));copies=np.zeros(4,np.int64);fetches=[]
 for event in range(65*48):
  lo,hi=a['offsets'][event:event+2];org=a['prefill_origins'] if event<48 else a['decode_origins'];sel=a['selected'][lo:hi]
  out,_,_=c.plan_current(event,sel,a['weights'][lo:hi],org,a['gates'][event])
  for rank,*_ in out[5]:copies[rank]+=1
  if budget==0:fetches.append(out[5])
  if budget and event>=48:
   hist=np.bincount((sel.astype(np.int64)+org.astype(np.int64)[:,None]*128).ravel(),minlength=512).reshape(4,128)
   for rank,*_ in c.plan_prefetch_next(hist):copies[rank]+=1
 c.arena.assert_consistent();assert not c.pending
 digest=lambda x:hashlib.sha256(x.tobytes()).hexdigest()
 d=dict(status='PASS',world=4,batch=batch,policy=policy,P=budget,horizon=64,rank_state_hashes=[digest(c.main.slots[r,:caps[r]]) for r in range(4)],rank_role_hashes=[digest(c.arena.main_physical[r]) for r in range(4)],counters=c.counters,max_copy_counts=copies.tolist())
 if budget==0:
  f=dst/f'{policy}_fetches.json';write(f,fetches);d.update(fetches_sha256=sha(f),H2D_bytes=(copies*9437184).tolist())
 write(target,d);return {k:d[k] for k in ('status','batch','policy','P')}
def main():
 ROOT.mkdir(exist_ok=True)
 for batch in (128,256):prepare(batch)
 specs=[(b,p,k) for b in (128,256) for p in ('BR','LA','CA') for k in (0,2)]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  for result in pool.map(proof,specs):print(result,flush=True)
 for batch in (128,256):
  dst=ROOT/f'inputs_B{batch}_H64';meta=json.loads((dst/'input_receipt.json').read_text())
  write(dst/'receipt.json',dict(status='PASS',world=4,batch=batch,horizon=64,winner={'placement_seed':meta['placement_seed']},proofs={p:json.loads((dst/f'{p}_P0_proof.json').read_text()) for p in ('BR','LA','CA')},source_receipt_sha256=sha(dst/'input_receipt.json')))
 write(PACKET/'R4_H64_CPU_PROOFS.json',dict(status='PASS',world=4,horizon=64,batches=[128,256],policies=['BR','LA','CA'],budgets=[0,2],proofs={str(p):sha(p) for p in ROOT.glob('inputs*/*_proof.json')}))
if __name__=='__main__':main()
