"""CPU-only R8/B128/c60 stress search and temporary replica oracle."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[k]='1'
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from numba import njit

from ca_stress_cpu import Pool,exact as reference_exact
from ca_stress_common import ROOT as POOL_ROOT, sha, write
from br_carep_cpu import balanced_assignment,choose_slot,place,EXPERT_BYTES,ROW_BYTES

WORLD=8
BATCH=128
CACHE=60
HORIZON=256
PROXY_EVENTS=48
SAMPLE_TOP=16
PAIR_TOP=8
TRIPLES_PER_DATASET=8
PACKET=Path(__file__).resolve().parents[1]/'experiments/r8_b128_c60_replication_stress_20261004'
OUT=Path('/home/hwlee/mgo-results/r8_b128_c60_replication_stress_20261004')
PLACEMENT_SEEDS=np.arange(256,dtype=np.int64)

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def sample_counts(routes,sample):
    out=np.zeros((len(routes),128),np.int32)
    for i in range(len(routes)):
        out[i]=np.bincount(np.asarray(routes[i,sample]).reshape(-1),minlength=128)
    return out

def hotness_score(counts):
    vals=[]
    for row in counts:
        active=row[row>0]
        if not len(active):continue
        quota=(len(active)+WORLD-1)//WORLD
        worst=np.sort(active)[-quota:].sum()
        mean=row.sum()/WORLD
        vals.append(worst/mean)
    return float(np.mean(vals))

def owners_for_seed(counts,seed):
    rng=np.random.default_rng(int(seed));owners=np.full((len(counts),128),-1,np.int8)
    crit=0
    for ev,row in enumerate(counts):
        active=np.flatnonzero(row>0);m=len(active)
        slots=np.concatenate([np.full(m//WORLD+(r<m%WORLD),r,np.int8) for r in range(WORLD)])
        order=rng.permutation(m);assigned=np.empty(m,np.int8);assigned[order]=slots
        owners[ev,active]=assigned
        loads=np.zeros(WORLD,np.int64)
        for e,r in zip(active,assigned):loads[r]+=row[e]
        crit+=int(loads.max())
    return owners,int(crit)

@njit(cache=True)
def peer_proxy(routes,sample,orders,owners,batch):
    result=np.zeros(len(orders),np.int64)
    for trial in range(len(orders)):
        total=0
        for ev in range(len(routes)):
            for pos in range(len(sample)):
                req=sample[orders[trial,pos]]
                rank=pos//batch
                mask=0
                for k in range(8):
                    e=routes[ev,req,k];dst=owners[ev,e]
                    if dst!=rank:
                        total+=1
                        mask|=1<<dst
                for r in range(8):total+=(mask>>r)&1
        result[trial]=total
    return result

@njit(cache=True)
def replay_diag(selected,offsets,origins0,origins,gates_by_event,capacities,horizon,randomized,seed):
    np.random.seed(seed)
    layers=48;experts=128;world=len(capacities);keys=layers*experts;events=(horizon+1)*layers
    slots=np.full((world,int(capacities.max())),-1,np.int32)
    owner=np.zeros(keys,np.int16);primary=np.full(keys,-1,np.int8)
    last=np.zeros((world,keys),np.int32);seen=np.zeros(keys,np.bool_);lost=np.zeros(keys,np.bool_)
    birth=np.full((world,keys),-1,np.int32);reuses=np.zeros((world,keys),np.int32);gates=np.zeros((layers,experts),np.float32)
    rank_rows=np.zeros((events,world),np.int32)
    peer_bytes=np.zeros(events,np.int64);hit_routes=np.zeros(events,np.int64);miss_experts=np.zeros(events,np.int32)
    local_oracle=np.zeros(events,np.int32);split_oracle=np.zeros(events,np.int32);ideal=np.zeros(events,np.int32);oracle_hot=np.zeros(events,np.int32)
    dummy=np.zeros(48,np.float64)
    for event in range(events):
        layer=event%layers;step=event//layers;tick=event+1;gates[layer]=gates_by_event[event]
        org=origins0 if step==0 else origins;lo=offsets[event];hi=offsets[event+1];n=hi-lo
        demand=np.zeros((experts,world),np.int32);active=np.zeros(experts,np.bool_)
        resident=owner[layer*experts:(layer+1)*experts]!=0
        for t in range(n):
            r=org[t]
            for k in range(8):
                e=selected[lo+t,k];demand[e,r]+=1;active[e]=True
                if resident[e]:hit_routes[event]+=1
        misses=np.flatnonzero(active&(~resident));miss_experts[event]=len(misses)
        assignment=balanced_assignment(demand,misses,world,randomized)
        for i in range(len(misses)):
            e=misses[i];r=assignment[i];key=layer*experts+e
            slot=choose_slot(r,layer,active,slots,capacities,last,gates,True,experts)
            assert slot>=0
            dummy[:]=0
            place(r,key,slot,False,tick,slots,owner,primary,last,seen,lost,birth,reuses,dummy)
        served=np.zeros((world,experts),np.bool_)
        remote_routes=0;dispatch_pairs=0
        for t in range(n):
            r=org[t];mask=0
            for k in range(8):
                e=selected[lo+t,k];key=layer*experts+e
                dst=r if owner[key]&(1<<r) else primary[key]
                assert dst>=0
                rank_rows[event,dst]+=1;served[dst,e]=True
                if dst!=r:
                    remote_routes+=1;mask|=1<<dst
            for r2 in range(world):dispatch_pairs+=(mask>>r2)&1
        peer_bytes[event]=(remote_routes+dispatch_pairs)*ROW_BYTES
        base=rank_rows[event].max();best_local=base;best_split=base;best_hot=0
        for e in range(experts):
            if not active[e]:continue
            key=layer*experts+e;p=primary[key]
            if p<0:continue
            total=0
            for r in range(world):total+=demand[e,r]
            for q in range(world):
                if q==p:continue
                local_shift=demand[e,q]
                if local_shift>0:
                    m=0
                    for r in range(world):
                        v=rank_rows[event,r]
                        if r==p:v-=local_shift
                        elif r==q:v+=local_shift
                        if v>m:m=v
                    if m<best_local:best_local=m
                lp=rank_rows[event,p];lq=rank_rows[event,q]
                center=(lp-lq)//2
                if center<0:center=0
                if center>total:center=total
                for delta in range(-2,3):
                    shift=center+delta
                    if shift<0 or shift>total:continue
                    m=0
                    for r in range(world):
                        v=rank_rows[event,r]
                        if r==p:v-=shift
                        elif r==q:v+=shift
                        if v>m:m=v
                    if m<best_split:
                        best_split=m;best_hot=total
        local_oracle[event]=best_local;split_oracle[event]=best_split;oracle_hot[event]=best_hot
        total_rows=rank_rows[event].sum();ideal[event]=(total_rows+world-1)//world
        for r in range(world):
            for e in range(experts):
                if served[r,e]:last[r,layer*experts+e]=tick
    return rank_rows,peer_bytes,hit_routes,miss_experts,local_oracle,split_oracle,ideal,oracle_hot,slots,owner,primary

def summarize(result):
    rank_rows,peer,hit,miss,local,split,ideal,hot=result[:8]
    sl=slice(48,None);rows=rank_rows[sl];mx=rows.max(1);mean=rows.mean(1)
    cv=np.where(mean>0,rows.std(1)/mean,0)
    raw=int(rows.sum());base=int(mx.sum())
    hotv=hot[sl][hot[sl]>0]
    return dict(
        decode_sum_max_rank_expert_rows=base,
        decode_peak_max_rank_expert_rows=int(mx.max()),
        decode_mean_rank_cv=float(cv.mean()),
        decode_peer_bytes=int(peer[sl].sum()),
        decode_H2D_bytes=int(miss[sl].sum())*EXPERT_BYTES,
        decode_exact_global_hit_rate=float(hit[sl].sum()/raw),
        oracle_local_peak_max_rank_rows=int(local[sl].max()),
        oracle_split_peak_max_rank_rows=int(split[sl].max()),
        oracle_local_sum_max_rank_rows=int(local[sl].sum()),
        oracle_local_reduction=float(1-local[sl].sum()/base),
        oracle_split_sum_max_rank_rows=int(split[sl].sum()),
        oracle_split_reduction=float(1-split[sl].sum()/base),
        oracle_ideal_sum_max_rank_rows=int(ideal[sl].sum()),
        oracle_improved_event_fraction=float(np.mean(split[sl]<mx)),
        oracle_hot_rows_p50=float(np.percentile(hotv,50)) if len(hotv) else 0.0,
        oracle_hot_rows_p90=float(np.percentile(hotv,90)) if len(hotv) else 0.0,
        oracle_hot_rows_p99=float(np.percentile(hotv,99)) if len(hotv) else 0.0,
        oracle_hot_rows_max=int(hotv.max()) if len(hotv) else 0,
    )

def state_hash(result):
    h=hashlib.sha256()
    for arr in result[-3:]:h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()

def exact(pool,dataset,sample_seed,dp_seed,placement_seed):
    order=pool.candidate(WORLD,BATCH,sample_seed,dp_seed);arrays=pool.pack(order,WORLD,BATCH)
    slots=48*128*CACHE//100
    cap=np.array([slots//WORLD+(r<slots%WORLD) for r in range(WORLD)],np.int64)
    br_raw=replay_diag(arrays['selected'],arrays['offsets'],arrays['prefill_origins'],arrays['decode_origins'],arrays['gates'],cap,HORIZON,True,placement_seed)
    ca_raw=replay_diag(arrays['selected'],arrays['offsets'],arrays['prefill_origins'],arrays['decode_origins'],arrays['gates'],cap,HORIZON,False,42)
    manifest=dict(route_sha256=hashlib.sha256(arrays['selected'].tobytes()).hexdigest(),gate_sha256=hashlib.sha256(arrays['gates'].tobytes()).hexdigest(),pool_receipt_sha256=sha(pool.path/'receipt.json'),dataset=dataset,world=WORLD,batch=BATCH,cache=CACHE,sample_seed=sample_seed,dp_seed=dp_seed,placement_seed=placement_seed,request_ids=order.tolist(),ranks=[order[r*BATCH:(r+1)*BATCH].tolist() for r in range(WORLD)])
    return dict(manifest=manifest,manifest_sha256=digest(manifest),BR={**summarize(br_raw), 'final_state_sha256':state_hash(br_raw)},CA={**summarize(ca_raw),'final_state_sha256':state_hash(ca_raw)})

def prescreen_dataset(name):
    pool=Pool(name);routes=np.asarray(pool.a['proxy_routes']);assert pool.n>=WORLD*BATCH
    sample_rows=[]
    cache={}
    for s in range(256):
        sample=np.random.default_rng(s).permutation(pool.n)[:WORLD*BATCH]
        counts=sample_counts(routes,sample);score=hotness_score(counts);cache[s]=(sample,counts)
        sample_rows.append(dict(sample_seed=s,hotness_score=score))
    sample_rows.sort(key=lambda x:(-x['hotness_score'],x['sample_seed']))
    top_samples=sample_rows[:SAMPLE_TOP]
    pairs=[]
    for row in top_samples:
        s=row['sample_seed'];sample,counts=cache[s]
        best=[]
        for ps in range(256):
            owners,crit=owners_for_seed(counts,ps);best.append((crit,ps,owners))
        best.sort(key=lambda x:(-x[0],x[1]))
        for crit,ps,owners in best[:2]:
            pairs.append(dict(dataset=name,sample_seed=s,placement_seed=ps,critical_proxy=crit,hotness_score=row['hotness_score'],owners=owners,sample=sample))
    pairs.sort(key=lambda x:(-x['critical_proxy'],-x['hotness_score'],x['sample_seed'],x['placement_seed']))
    pairs=pairs[:PAIR_TOP]
    triples=[]
    orders=np.array([np.random.default_rng(dp).permutation(WORLD*BATCH) for dp in range(256)],np.int64)
    for pair in pairs:
        scores=peer_proxy(routes,pair['sample'],orders,pair['owners'],BATCH)
        order=np.argsort(-scores,kind='stable')[:2]
        for dp in order:
            triples.append(dict(dataset=name,sample_seed=pair['sample_seed'],placement_seed=pair['placement_seed'],dp_seed=int(dp),critical_proxy=pair['critical_proxy'],peer_proxy_rows=int(scores[dp]),hotness_score=pair['hotness_score']))
    triples.sort(key=lambda x:(-x['critical_proxy'],-x['peer_proxy_rows'],-x['hotness_score'],x['sample_seed'],x['placement_seed'],x['dp_seed']))
    return sample_rows,triples[:TRIPLES_PER_DATASET]

def validate_reference():
    """Require exact resource parity with the completed R8/B64/c60 replay."""
    pool=Pool('MATH');world=8;batch=64;sample_seed=8;dp_seed=228
    order=pool.candidate(world,batch,sample_seed,dp_seed);arrays=pool.pack(order,world,batch)
    slots=48*128*60//100;cap=np.array([slots//world+(r<slots%world) for r in range(world)],np.int64)
    rows=[]
    for policy,randomized,seed in [('BR',True,42),('CA',False,42)]:
        diag=replay_diag(arrays['selected'],arrays['offsets'],arrays['prefill_origins'],arrays['decode_origins'],arrays['gates'],cap,256,randomized,seed)
        ref=reference_exact(arrays,world,60,policy,seed)
        rank_rows,peer,hit,miss=diag[:4]
        raw=int(rank_rows.sum());peer_bytes=int(peer.sum());h2d=int(miss.sum())*EXPERT_BYTES;hit_rate=float(hit.sum()/raw)
        assert peer_bytes==ref['full']['peer_bytes'],(policy,'peer',peer_bytes,ref['full']['peer_bytes'])
        assert h2d==ref['full']['H2D_bytes'],(policy,'H2D',h2d,ref['full']['H2D_bytes'])
        assert abs(hit_rate-ref['full']['exact_global_hits_fraction'])<1e-12,(policy,'hit',hit_rate,ref['full']['exact_global_hits_fraction'])
        rows.append(dict(policy=policy,peer_bytes=peer_bytes,H2D_bytes=h2d,exact_global_hit_rate=hit_rate,reference_state_sha256=ref['final_state_sha256']))
    receipt=dict(status='PASS',validation_kind='fresh unchanged reference implementation replay; not an archived receipt comparison',reference_source_sha256=sha(Path(__file__).with_name('ca_stress_cpu.py')),reference='MATH R8/B64/c60 sample8/dp228 seed42',rows=rows)
    write(PACKET/'implementation_validation.json',receipt);return receipt

def main():
    OUT.mkdir(parents=True,exist_ok=True);started=time.time()
    validate_reference()
    stage_a={};retained=[]
    for dataset in ('MATH','ShareGPT'):
        samples,triples=prescreen_dataset(dataset);stage_a[dataset]=dict(top_samples=samples[:SAMPLE_TOP],retained_triples=triples);retained.extend(triples)
    write(OUT/'stage_a.json',dict(status='PASS',stage_a=stage_a))
    exact_rows=[]
    for i,triple in enumerate(retained):
        pool=Pool(triple['dataset']);row=exact(pool,**{k:triple[k] for k in ('dataset','sample_seed','dp_seed','placement_seed')});row['proxy']=triple;exact_rows.append(row)
        write(OUT/f'exact_{i:02d}.json',row)
        print(json.dumps(dict(done=i+1,total=len(retained),dataset=triple['dataset'],sample=triple['sample_seed'],dp=triple['dp_seed'],placement=triple['placement_seed'],BR=row['BR']['decode_sum_max_rank_expert_rows'])),flush=True)
    exact_rows.sort(key=lambda x:(-x['BR']['decode_sum_max_rank_expert_rows'],-x['BR']['decode_peer_bytes'],-x['BR']['decode_peak_max_rank_expert_rows'],x['manifest']['dataset'],x['manifest']['sample_seed'],x['manifest']['dp_seed'],x['manifest']['placement_seed']))
    winner=exact_rows[0]
    top=[dict(position=i+1,manifest=r['manifest'],BR=r['BR'],CA=r['CA']) for i,r in enumerate(exact_rows[:5])]
    result=dict(status='PASS',winner=winner,top5=top,selection='BR-only: max decode sum max-rank rows; tie peer bytes; tie peak max-rank rows',oracle_gate=dict(threshold=0.10,BR_pass=winner['BR']['oracle_split_reduction']>=.10,CA_pass=winner['CA']['oracle_split_reduction']>=.10),source_pool_root=str(POOL_ROOT),seconds=time.time()-started)
    write(OUT/'RESULT.json',result);write(PACKET/'RESULT.json',result)
    write(PACKET/'selected_manifest.json',winner['manifest'])
    lines=['# R8/B128/c60 replication stress result','',f"Winner: {winner['manifest']['dataset']} sample={winner['manifest']['sample_seed']} dp={winner['manifest']['dp_seed']} placement={winner['manifest']['placement_seed']}",'',
      '| Policy | sum max-rank rows | peak max-rank rows | peer GiB | H2D TiB | exact hit | 1rep split reduction |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('BR','CA'):
        x=winner[name];lines.append(f"| {name} | {x['decode_sum_max_rank_expert_rows']} | {x['decode_peak_max_rank_expert_rows']} | {x['decode_peer_bytes']/2**30:.3f} | {x['decode_H2D_bytes']/2**40:.3f} | {x['decode_exact_global_hit_rate']:.2%} | {x['oracle_split_reduction']:.2%} |")
    lines+=['',f"Oracle gate: BR={result['oracle_gate']['BR_pass']}, CA={result['oracle_gate']['CA_pass']}.",'Temporary replica is an upper bound only; no physical timing claim.']
    (PACKET/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='PASS',winner=winner['manifest'],BR_oracle=winner['BR']['oracle_split_reduction'],CA_oracle=winner['CA']['oracle_split_reduction'])),flush=True)

if __name__=='__main__':main()
