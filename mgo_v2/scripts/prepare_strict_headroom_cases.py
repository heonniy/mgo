"""Prepare exact strict-phase BR/LA/LA_CA_NEAR physical inputs for S0 triples."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import hashlib,json
import numpy as np
from strict_headroom_common import *
import mgo_v2  # Initialize package before legacy controller imports.
from env_offload_policy import Policy

EB=9437184
POLICIES=(('BR',0),('LA',4),('LA_CA_NEAR',7))

def array_hash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def prepare(row):
    import psutil
    assert psutil.virtual_memory().available>256*2**30, 'host memory guard'
    world=int(row['world']);batch=int(row['batch']);context=int(row['context'])
    sample=int(row['sample_seed']);dp=int(row['dp_seed']);br=int(row['br_seed'])
    label=f'R{world}_B{batch}_L{context}_s{sample}_d{dp}_r{br}'
    dst=ROOT/'inputs'/label
    if (dst/'receipt.json').exists():
        assert json.loads((dst/'receipt.json').read_text())['status']=='PASS';return str(dst)
    dst.mkdir(parents=True,exist_ok=True)
    ranks=ranked_ids(world,batch,sample,dp);flat_ids=ranks.reshape(-1);n=world*batch*context
    pool=pool_dir(context)
    selected=np.load(pool/'selected.npy',mmap_mode='r');weights=np.load(pool/'weights.npy',mmap_mode='r')
    gates_pool=np.load(pool/'gates.npy',mmap_mode='r')
    out_s=np.lib.format.open_memmap(dst/'selected.npy',mode='w+',shape=(48*n,8),dtype='uint8')
    out_w=np.lib.format.open_memmap(dst/'weights.npy',mode='w+',shape=(48*n,8),dtype='float32')
    for layer in range(48):
        out_s[layer*n:(layer+1)*n]=selected[flat_ids,layer].reshape(n,8)
        out_w[layer*n:(layer+1)*n]=weights[flat_ids,layer].reshape(n,8)
    out_s.flush();out_w.flush()
    gates=np.asarray(gates_pool[flat_ids],np.float64).mean(0).astype(np.float32);np.save(dst/'gates.npy',gates)
    np.save(dst/'offsets.npy',np.arange(49,dtype=np.int64)*n)
    np.save(dst/'prefill_origins.npy',np.repeat(np.arange(world,dtype=np.int8),batch*context))
    np.save(dst/'decode_origins.npy',np.repeat(np.arange(world,dtype=np.int8),batch))
    source=request_rows();req_ranks=[]
    for rr in ranks:
        part=[]
        for q in rr:
            x=dict(source[int(q)]);x['input_ids']=x['input_ids'][-context:];part.append(x)
        req_ranks.append(part)
    write(dst/'requests.json',dict(status='PASS',world=world,batch=batch,context=context,ranks=req_ranks,
        request_ids=ranks.tolist()))
    total_slots=48*128*30//100
    caps=[total_slots//world+(r<total_slots%world) for r in range(world)]
    arrays={k:np.load(dst/(k+'.npy'),mmap_mode='r') for k in ('selected','weights','offsets','prefill_origins','gates')}
    proofs={}
    for policy,kind in POLICIES:
        seed=br if policy=='BR' else 0
        c=Policy(caps,np.zeros((48,128,128),np.float32),False,kind,seed)
        fetches=[];copies=np.zeros(world,np.int64);layer_counts=[]
        for layer in range(48):
            lo,hi=arrays['offsets'][layer:layer+2]
            out=c.apply(layer,arrays['selected'][lo:hi],arrays['weights'][lo:hi],
                arrays['prefill_origins'],arrays['gates'][layer],np.zeros((128,world),np.int32))
            fs=[[int(v) for v in x] for x in out[5]];fetches.append(fs)
            counts=np.bincount([x[0] for x in fs],minlength=world);copies+=counts;layer_counts.append(counts.tolist())
            assert counts.max()-counts.min()<=1 and all(x[-1]==0 for x in fs)
        fp=dst/f'{policy}_fetches.json';write(fp,fetches)
        proofs[policy]=dict(status='PASS',
            rank_state_hashes=[array_hash(c.slots[r,:caps[r]]) for r in range(world)],
            H2D_bytes=(copies*EB).tolist(),copies=copies.tolist(),layer_fetch_counts=layer_counts,
            fetches_sha256=sha(fp),substitution=False,replication=False)
    write(dst/'receipt.json',dict(status='PASS',cache=30,world=world,batch=batch,context=context,
        sample_seed=sample,dp_seed=dp,br_seed=br,placement_seed=br,capacities=caps,proofs=proofs,
        request_manifest_sha256=sha(OLD_ROOT/'requests.json')))
    return str(dst)

def main():
    rows=json.loads((PACKET/'STRICT_HEADROOM_S0.json').read_text())['rows']
    paths=[]
    import concurrent.futures,multiprocessing
    with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        for p in pool.map(prepare,rows):
            paths.append(p);print('prepared',p,flush=True)
    write(PACKET/'STRICT_HEADROOM_INPUTS.json',dict(status='PASS',count=len(paths),paths=paths))

if __name__=='__main__':main()
