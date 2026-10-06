"""BR-adversarial CPU seed search for strict-phase LA_CA_NEAR.

Searches sample_seed x dp_seed x BR random seed specifically to maximize
LA_CA_NEAR headroom over BR.  This is an oracle/stress search, not average-case.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import json,time
from pathlib import Path
import numpy as np
from numba import njit
from strict_headroom_common import *
from br_carep_cpu import balanced_assignment
from env_offload_policy import seed_rng
from la_placement import load_locality_near_assignment

SAMPLE_SEEDS=256
DP_SEEDS=64
BR_SEEDS=64
KEEP_SAMPLES=8
KEEP_TRIPLES=8

def ensure_l256_pool():
    dst=ROOT/'pool_L256';receipt=dst/'receipt.json'
    if receipt.exists():
        assert json.loads(receipt.read_text())['status']=='PASS';return
    dst.mkdir(parents=True,exist_ok=True)
    shapes={
      'selected':((2048,48,256,8),'uint8'),
      'weights':((2048,48,256,8),'float32'),
      'gates':((2048,48,128),'float32'),
      'first_tokens':((2048,),'int64'),
    }
    outs={k:np.lib.format.open_memmap(dst/(k+'.npy'),mode='w+',shape=s,dtype=d) for k,(s,d) in shapes.items()}
    for i,g in enumerate(R4_GPUS):
        src=ROOT/'capture_L256'/f'gpu{g}';r=json.loads((src/'receipt.json').read_text())
        assert r['status']=='PASS' and r['request_ids']==list(range(i*512,(i+1)*512))
        for k in outs:
            p=src/(k+'.npy');assert sha(p)==r['files'][p.name]
            outs[k][i*512:(i+1)*512]=np.load(p,mmap_mode='r')
    for x in outs.values():x.flush()
    write(receipt,dict(status='PASS',context=256,request_manifest_sha256=sha(OLD_ROOT/'requests.json'),
        files={k:sha(dst/(k+'.npy')) for k in outs}))

@njit(cache=True)
def build_hist(selected):
    n,l,t,k=selected.shape
    out=np.zeros((n,l,128),np.int32)
    for q in range(n):
        for layer in range(l):
            for i in range(t):
                for j in range(k):
                    out[q,layer,selected[q,layer,i,j]]+=1
    return out

def hist_for(context):
    path=ROOT/f'hist_L{context}.npy'
    if path.exists():return np.load(path,mmap_mode='r')
    src=np.load(pool_dir(context)/'selected.npy',mmap_mode='r')
    out=build_hist(src);np.save(path,out);return np.load(path,mmap_mode='r')

def sample_score(hist,ids):
    d=np.asarray(hist[ids],np.int64).sum(0)
    total=d.sum(1).clip(min=1)
    # Random-placement rank-load variance grows with expert-demand squared mass.
    return float(np.sum((d.astype(np.float64)**2).sum(1)/total))

def demands_for(hist,rank_ids):
    world,batch=rank_ids.shape
    d=np.zeros((48,128,world),np.int64)
    for r in range(world):d[:,:,r]=np.asarray(hist[rank_ids[r]],np.int64).sum(0)
    return d

@njit(cache=True)
def assignment_metrics(demand,active,assignment,world):
    rows=np.zeros(world,np.int64);remote=0
    for i in range(len(active)):
        e=int(active[i]);dst=int(assignment[i]);total=0
        for r in range(world):total+=demand[e,r]
        rows[dst]+=total;remote+=total-demand[e,dst]
    mx=0
    for r in range(world):
        if rows[r]>mx:mx=rows[r]
    return mx,remote

@njit(cache=True)
def evaluate_dp(demands,world):
    owner=np.zeros(48*128,np.int16)
    cand_crit=0;cand_remote=0
    candidate_assign=np.full((48,128),-1,np.int8)
    active_by_layer=np.zeros((48,128),np.bool_)
    for layer in range(48):
        total=demands[layer].sum(1);active=np.flatnonzero(total>0);active_by_layer[layer,active]=True
        a=load_locality_near_assignment(demands[layer],active,owner,layer,200)
        mx,remote=assignment_metrics(demands[layer],active,a,world);cand_crit+=mx;cand_remote+=remote
        for i in range(len(active)):candidate_assign[layer,active[i]]=a[i]
    br_crit=np.zeros(BR_SEEDS,np.int64);br_remote=np.zeros(BR_SEEDS,np.int64)
    for seed in range(BR_SEEDS):
        seed_rng(seed)
        for layer in range(48):
            active=np.flatnonzero(active_by_layer[layer])
            a=balanced_assignment(demands[layer],active,world,True)
            mx,remote=assignment_metrics(demands[layer],active,a,world)
            br_crit[seed]+=mx;br_remote[seed]+=remote
    return cand_crit,cand_remote,br_crit,br_remote

def main():
    ROOT.mkdir(parents=True,exist_ok=True);ensure_l256_pool()
    # Compile the hot functions once before the large search.
    dummy=np.ones((48,128,4),np.int64);evaluate_dp(dummy,4)
    rows=[];cell_summary=[]
    for context in CONTEXTS:
        hist=hist_for(context)
        for world in WORLDS:
            for batch in BATCHES:
                n=world*batch;assert n<=512
                potentials=[]
                for sample_seed in range(SAMPLE_SEEDS):
                    ids=sample_ids(world,batch,sample_seed)
                    potentials.append((sample_score(hist,ids),sample_seed))
                potentials.sort(reverse=True);kept=[s for _,s in potentials[:KEEP_SAMPLES]]
                candidates=[]
                started=time.time()
                for sample_seed in kept:
                    ids=sample_ids(world,batch,sample_seed)
                    for dp_seed in range(DP_SEEDS):
                        shuffled=np.random.default_rng(dp_seed).permutation(ids).reshape(world,batch)
                        demands=demands_for(hist,shuffled)
                        cc,cr,bc,br=evaluate_dp(demands,world)
                        for br_seed in range(BR_SEEDS):
                            load_gain=1-float(cc)/float(bc[br_seed])
                            comm_gain=1-float(cr)/float(br[br_seed]) if br[br_seed] else 0.0
                            candidates.append(dict(world=world,batch=batch,context=context,
                                sample_seed=sample_seed,dp_seed=dp_seed,br_seed=br_seed,
                                load_proxy_gain=load_gain,remote_route_proxy_gain=comm_gain,
                                BR=dict(sum_max_rank_rows=int(bc[br_seed]),remote_expert_routes=int(br[br_seed])),
                                candidate=dict(sum_max_rank_rows=int(cc),remote_expert_routes=int(cr))))
                candidates.sort(key=lambda x:(-x['load_proxy_gain'],-x['remote_route_proxy_gain'],
                    x['sample_seed'],x['dp_seed'],x['br_seed']))
                top=candidates[:KEEP_TRIPLES];rows.extend(top)
                cell_summary.append(dict(world=world,batch=batch,context=context,
                    retained_sample_seeds=kept,best=top[0],seconds=time.time()-started))
                write(ROOT/'screen_progress.json',dict(status='RUNNING',rows=rows,cells=cell_summary))
    out=dict(status='PASS',purpose='maximum BR-adversarial headroom; NOT average-case',
        sample_seed_range=[0,SAMPLE_SEEDS-1],dp_seed_range=[0,DP_SEEDS-1],br_seed_range=[0,BR_SEEDS-1],
        ranking='maximize critical-rank expert-row reduction vs BR, then remote expert-route reduction',
        near_tie_bps=200,rows=rows,cells=cell_summary)
    write(PACKET/'STRICT_HEADROOM_S0.json',out);write(ROOT/'screen_status.json',out)

if __name__=='__main__':main()
