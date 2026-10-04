"""Phase-aware CPU replay for R8 B128/B256 placement + persistent replication.

This is a resource/headroom simulator. It never claims physical TPOT/E2E.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[_k]="1"

import csv
import hashlib
import itertools
import json
import math
import resource
import statistics
import time
from pathlib import Path

import numpy as np
from numba import njit

from ca_stress_cpu import Pool
from ca_stress_common import write, sha
from br_carep_cpu import balanced_assignment, choose_slot, place, EXPERT_BYTES, ROW_BYTES

WORLD=8
CACHE_PERCENT=60
LAYERS=48
EXPERTS=128
TOPK=8
HORIZON=256
PKG=Path(__file__).resolve().parents[1]
PACKET=PKG/"experiments/r8_real_replica_batch_scaling_20261004"
OUT=Path("/home/hwlee/mgo-results/r8_real_replica_batch_scaling_20261004")
OLD_B128=PKG/"experiments/r8_b128_c60_replication_stress_20261004/RESULT.json"

# policy: 0 BR, 1 CA, 2 LA
POLICY_NAMES={0:"BR",1:"CA",2:"LA"}

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()

@njit(cache=True)
def bitcount8(x):
    c=0
    for b in range(8):
        c += (x>>b)&1
    return c

@njit(cache=True)
def first_owner(mask):
    for r in range(8):
        if mask&(1<<r):
            return r
    return -1

@njit(cache=True)
def second_owner(mask, first):
    for r in range(8):
        if r!=first and mask&(1<<r):
            return r
    return -1

@njit(cache=True)
def estimate_loads(demand, owner, layer):
    loads=np.zeros(WORLD,np.int64)
    totals=demand.sum(1)
    order=np.argsort(-totals)
    base=layer*EXPERTS
    for ii in range(EXPERTS):
        e=order[ii]
        total=totals[e]
        if total==0:
            continue
        mask=owner[base+e]
        if mask==0:
            continue
        p=first_owner(mask)
        if bitcount8(mask)==1:
            loads[p]+=total
        else:
            q=second_owner(mask,p)
            lp=loads[p]+total
            lq=loads[q]
            shift=(lp-lq)//2
            if shift<0: shift=0
            if shift>total: shift=total
            loads[p]+=total-shift
            loads[q]+=shift
    return loads

@njit(cache=True)
def load_assignment(demand, misses, owner, layer):
    """Balanced miss-count quota, greedy min-max execution-load placement."""
    m=len(misses)
    quota=np.empty(WORLD,np.int64)
    for r in range(WORLD):
        quota[r]=m//WORLD+(r<m%WORLD)
    remaining=quota.copy()
    loads=estimate_loads(demand,owner,layer)
    totals=np.empty(m,np.int64)
    for i in range(m):
        totals[i]=demand[misses[i]].sum()
    order=np.argsort(-totals)
    answer=np.empty(m,np.int64)
    for oi in range(m):
        i=order[oi]
        e=misses[i]
        total=totals[i]
        best=-1
        best_max=2**60
        best_dst_load=2**60
        best_local=-1
        for r in range(WORLD):
            if remaining[r]<=0:
                continue
            mx=0
            for rr in range(WORLD):
                v=loads[rr]+(total if rr==r else 0)
                if v>mx: mx=v
            dstload=loads[r]+total
            local=demand[e,r]
            if (mx<best_max or
                (mx==best_max and dstload<best_dst_load) or
                (mx==best_max and dstload==best_dst_load and local>best_local) or
                (mx==best_max and dstload==best_dst_load and local==best_local and (best<0 or r<best))):
                best=r;best_max=mx;best_dst_load=dstload;best_local=local
        assert best>=0
        answer[i]=best
        remaining[best]-=1
        loads[best]+=total
    assert remaining.sum()==0
    return answer

@njit(cache=True)
def count_duplicate_copies(owner):
    copies=0
    unique=0
    for k in range(len(owner)):
        c=bitcount8(owner[k])
        copies+=c
        if c: unique+=1
    return copies-unique

@njit(cache=True)
def route_with_replicas(selected, lo, n, origins, layer, owner, primary):
    """Return rank loads and exact route destinations.

    Start from primary owner, then use the second copy to reduce current rank
    load. Rows local to the replica rank are moved first as a locality tie-break.
    """
    dest=np.empty((n,TOPK),np.int8)
    loads=np.zeros(WORLD,np.int64)
    totals=np.zeros(EXPERTS,np.int32)
    base=layer*EXPERTS
    for t in range(n):
        for k in range(TOPK):
            e=selected[lo+t,k]
            totals[e]+=1
            p=primary[base+e]
            assert p>=0
            dest[t,k]=p
            loads[p]+=1

    order=np.argsort(-totals)
    for oi in range(EXPERTS):
        e=order[oi]
        total=totals[e]
        if total<=0: continue
        mask=owner[base+e]
        if bitcount8(mask)!=2: continue
        p=primary[base+e]
        q=second_owner(mask,p)
        if q<0: continue
        shift=(loads[p]-loads[q])//2
        if shift<0: shift=0
        if shift>total: shift=total
        moved=0
        for t in range(n):
            if moved>=shift: break
            if origins[t]!=q: continue
            for k in range(TOPK):
                if moved>=shift: break
                if selected[lo+t,k]==e and dest[t,k]==p:
                    dest[t,k]=q;loads[p]-=1;loads[q]+=1;moved+=1
        for t in range(n):
            if moved>=shift: break
            for k in range(TOPK):
                if moved>=shift: break
                if selected[lo+t,k]==e and dest[t,k]==p:
                    dest[t,k]=q;loads[p]-=1;loads[q]+=1;moved+=1
    return dest,loads

@njit(cache=True)
def replay_policy(selected, offsets, origins0, origins, gates_by_event, capacities,
                  horizon, policy_kind, placement_seed,
                  replica_enabled, hot_mult, min_gain, dup_budget_frac,
                  local_batch, replica_all_phase):
    np.random.seed(placement_seed)
    keys=LAYERS*EXPERTS
    events=(horizon+1)*LAYERS
    slots=np.full((WORLD,int(capacities.max())),-1,np.int32)
    owner=np.zeros(keys,np.int16)
    primary=np.full(keys,-1,np.int8)
    last=np.zeros((WORLD,keys),np.int32)
    seen=np.zeros(keys,np.bool_)
    lost=np.zeros(keys,np.bool_)
    birth=np.full((WORLD,keys),-1,np.int32)
    reuses=np.zeros((WORLD,keys),np.int32)
    gates=np.zeros((LAYERS,EXPERTS),np.float32)

    max_rows=np.zeros(events,np.int32)
    total_rows=np.zeros(events,np.int32)
    rank_cv=np.zeros(events,np.float64)
    peer_dispatch_pairs=np.zeros(events,np.int64)
    peer_return_routes=np.zeros(events,np.int64)
    max_dispatch_pair_rows=np.zeros(events,np.int32)
    max_return_pair_rows=np.zeros(events,np.int32)
    active_peer_pairs=np.zeros(events,np.int16)
    h2d_counts=np.zeros((events,WORLD),np.int16)
    global_hit_routes=np.zeros(events,np.int64)
    raw_routes=np.zeros(events,np.int64)
    unique_misses=np.zeros(events,np.int16)
    first_fetches=np.zeros(events,np.int16)
    reload_fetches=np.zeros(events,np.int16)
    replica_created=np.zeros(events,np.int8)
    replica_source_miss=np.zeros(events,np.int8)
    replica_source_rank=np.full(events,-1,np.int8)
    replica_dest_rank=np.full(events,-1,np.int8)
    replica_served_rows=np.zeros(events,np.int32)
    victim_unique=np.zeros(events,np.int8)
    victim_duplicate=np.zeros(events,np.int8)
    duplicate_copies=np.zeros(events,np.int16)
    evictions=np.zeros(events,np.int16)

    dummy=np.zeros(48,np.float64)
    total_slots=int(capacities.sum())
    dup_budget=max(1,int(total_slots*dup_budget_frac))

    for event in range(events):
        layer=event%LAYERS
        step=event//LAYERS
        tick=event+1
        gates[layer]=gates_by_event[event]
        org=origins0 if step==0 else origins
        lo=offsets[event]
        hi=offsets[event+1]
        n=hi-lo
        assert n==len(org)

        demand=np.zeros((EXPERTS,WORLD),np.int32)
        active=np.zeros(EXPERTS,np.bool_)
        resident_before=np.zeros(EXPERTS,np.bool_)
        base=layer*EXPERTS
        for e in range(EXPERTS):
            resident_before[e]=owner[base+e]!=0

        for t in range(n):
            r=org[t]
            for k in range(TOPK):
                e=selected[lo+t,k]
                demand[e,r]+=1
                active[e]=True
                raw_routes[event]+=1
                if resident_before[e]:
                    global_hit_routes[event]+=1

        misses=np.flatnonzero(active & (~resident_before))
        unique_misses[event]=len(misses)
        if policy_kind==0:
            assignment=balanced_assignment(demand,misses,WORLD,True)
        elif policy_kind==1:
            assignment=balanced_assignment(demand,misses,WORLD,False)
        else:
            assignment=load_assignment(demand,misses,owner,layer)

        newly_missed=np.zeros(EXPERTS,np.bool_)
        row=np.zeros(48,np.float64)
        for i in range(len(misses)):
            e=misses[i]
            r=assignment[i]
            h2d_counts[event,r]+=1
            key=base+e
            slot=choose_slot(r,layer,active,slots,capacities,last,gates,True,EXPERTS)
            assert slot>=0
            victim=slots[r,slot]
            if victim>=0:
                evictions[event]+=1
            was_seen=seen[key]
            place(r,key,slot,False,tick,slots,owner,primary,last,seen,lost,birth,reuses,row)
            if was_seen: reload_fetches[event]+=1
            else: first_fetches[event]+=1
            newly_missed[e]=True

        allow_replica=replica_enabled and (replica_all_phase or step>0)
        if allow_replica:
            loads=estimate_loads(demand,owner,layer)
            base_max=loads.max()
            totals=demand.sum(1)
            dup_now=count_duplicate_copies(owner)
            best_gain=0
            best_e=-1;best_q=-1;best_slot=-1;best_victim=-1;best_p=-1
            best_victim_is_dup=False
            hot_threshold=hot_mult*local_batch
            if base_max>0:
                order=np.argsort(-totals)
                for oi in range(EXPERTS):
                    e=order[oi]
                    total=totals[e]
                    if total<hot_threshold or total<=0:
                        continue
                    key=base+e
                    mask=owner[key]
                    if bitcount8(mask)!=1:
                        continue
                    p=first_owner(mask)
                    for q in range(WORLD):
                        if q==p: continue
                        slot=choose_slot(q,layer,active,slots,capacities,last,gates,True,EXPERTS)
                        if slot<0: continue
                        victim=slots[q,slot]
                        victim_is_dup=False
                        extra_dup=1
                        if victim>=0:
                            victim_is_dup=bitcount8(owner[victim])>=2
                            if victim_is_dup: extra_dup=0
                        if dup_now+extra_dup>dup_budget:
                            continue
                        shift=(loads[p]-loads[q])//2
                        if shift<0: shift=0
                        if shift>total: shift=total
                        if shift<=0: continue
                        newmax=0
                        for rr in range(WORLD):
                            v=loads[rr]
                            if rr==p: v-=shift
                            elif rr==q: v+=shift
                            if v>newmax: newmax=v
                        gain=base_max-newmax
                        if gain<=0: continue
                        frac=gain/base_max
                        if frac<min_gain: continue
                        if (gain>best_gain or
                            (gain==best_gain and total>(totals[best_e] if best_e>=0 else -1)) or
                            (gain==best_gain and total==(totals[best_e] if best_e>=0 else -1) and q<best_q)):
                            best_gain=gain;best_e=e;best_q=q;best_slot=slot
                            best_victim=victim;best_p=p;best_victim_is_dup=victim_is_dup
            if best_e>=0:
                key=base+best_e
                if best_victim>=0:
                    if best_victim_is_dup: victim_duplicate[event]=1
                    else: victim_unique[event]=1
                    evictions[event]+=1
                place(best_q,key,best_slot,True,tick,slots,owner,primary,last,seen,lost,birth,reuses,row)
                replica_created[event]=1
                replica_source_rank[event]=best_p
                replica_dest_rank[event]=best_q
                replica_source_miss[event]=1 if newly_missed[best_e] else 0

        dest,loads=route_with_replicas(selected,lo,n,org,layer,owner,primary)
        total_rows[event]=loads.sum()
        max_rows[event]=loads.max()
        mean=loads.mean()
        rank_cv[event]=loads.std()/mean if mean>0 else 0.0

        dispatch_pair=np.zeros((WORLD,WORLD),np.int32)
        return_pair=np.zeros((WORLD,WORLD),np.int32)
        served=np.zeros((WORLD,EXPERTS),np.bool_)
        for t in range(n):
            origin=org[t]
            mask=0
            for k in range(TOPK):
                e=selected[lo+t,k]
                dst=dest[t,k]
                served[dst,e]=True
                if bitcount8(owner[base+e])>=2 and dst!=primary[base+e]:
                    replica_served_rows[event]+=1
                if dst!=origin:
                    peer_return_routes[event]+=1
                    return_pair[dst,origin]+=1
                    mask |= 1<<dst
            for dst in range(WORLD):
                if mask&(1<<dst):
                    peer_dispatch_pairs[event]+=1
                    dispatch_pair[origin,dst]+=1

        md=0;mr=0;ap=0
        for a in range(WORLD):
            for b in range(WORLD):
                if dispatch_pair[a,b]>md: md=dispatch_pair[a,b]
                if return_pair[a,b]>mr: mr=return_pair[a,b]
                if dispatch_pair[a,b]>0 or return_pair[a,b]>0: ap+=1
        max_dispatch_pair_rows[event]=md
        max_return_pair_rows[event]=mr
        active_peer_pairs[event]=ap

        for r in range(WORLD):
            for e in range(EXPERTS):
                if served[r,e]:
                    key=base+e
                    if birth[r,key]>=0 and birth[r,key]<tick:
                        reuses[r,key]+=1
                    last[r,key]=tick

        duplicate_copies[event]=count_duplicate_copies(owner)

    return (max_rows,total_rows,rank_cv,peer_dispatch_pairs,peer_return_routes,
            max_dispatch_pair_rows,max_return_pair_rows,active_peer_pairs,
            h2d_counts,global_hit_routes,raw_routes,unique_misses,first_fetches,
            reload_fetches,replica_created,replica_source_miss,replica_source_rank,
            replica_dest_rank,replica_served_rows,victim_unique,victim_duplicate,
            duplicate_copies,evictions,slots,owner,primary)

RESULT_NAMES=[
    "max_rows","total_rows","rank_cv","peer_dispatch_pairs","peer_return_routes",
    "max_dispatch_pair_rows","max_return_pair_rows","active_peer_pairs",
    "h2d_counts","global_hit_routes","raw_routes","unique_misses","first_fetches",
    "reload_fetches","replica_created","replica_source_miss","replica_source_rank",
    "replica_dest_rank","replica_served_rows","victim_unique","victim_duplicate",
    "duplicate_copies","evictions","slots","owner","primary"
]

def as_result_dict(result):
    return dict(zip(RESULT_NAMES,result))

def scope_slice(name,horizon):
    if name=="prefill": return slice(0,LAYERS)
    if name=="decode": return slice(LAYERS,(horizon+1)*LAYERS)
    if name=="moe_total": return slice(0,(horizon+1)*LAYERS)
    raise KeyError(name)

def load_calibration():
    hot=json.loads((PKG/"experiments/hot_expert_replication_threshold_20261003/break_even_microcost.json").read_text())
    pair={}
    for row in hot["rows"]:
        if row.get("kind")=="pair-pooled" and row.get("mode")=="T0":
            pair[int(row["case"])]=float(row["median_ms"])
    h2d_path=PKG/"experiments/timing_stability_numa_20261004/S2B_H2D.json"
    h2d_medians=[];h2d_p90=[]
    if h2d_path.exists() and h2d_path.stat().st_size:
        try:
            obj=json.loads(h2d_path.read_text())
            rows=obj if isinstance(obj,list) else obj.get("rows",obj.get("records",[]))
            for row in rows:
                if (row.get("kind")=="H2D" and int(row.get("payload_bytes",0))==EXPERT_BYTES
                    and int(row.get("concurrency",0))==8):
                    h2d_medians.append(float(row["median_ms"]))
                    h2d_p90.append(float(row["p90_ms"]))
        except Exception:
            pass
    if not h2d_medians:
        tr=list(csv.DictReader((PKG/"experiments/fetch_comm_pareto_p2p_20261002/transport_calibration.csv").open()))
        row=next(r for r in tr if r["mode"]=="T0" and r["case"]=="h2d")
        h2d_medians=[float(row["h2d_median_ms"])]
        h2d_p90=[float(row["h2d_p90_ms"])]

    phase=json.loads((PKG/"experiments/rank_demand_oracle_20261002/gpu_phase_summary.json").read_text())
    loads=json.loads((PKG/"experiments/rank_demand_oracle_20261002/rank_load_summary.json").read_text())
    p0b8=next(r for r in loads if r["batch"]==8 and r["policy"]=="P0")
    denom=float(p0b8["decode_sum_max_rank_expert_rows"])
    gemm_slope=float(phase["P0"]["max_rank_gemm_gpu_union_ms"])/denom
    expert_slope=float(phase["P0"]["max_rank_expert_gpu_union_ms"])/denom

    tr=list(csv.DictReader((PKG/"experiments/fetch_comm_pareto_p2p_20261002/transport_calibration.csv").open()))
    peer=next(r for r in tr if r["mode"]=="T0" and r["case"]=="concurrent")
    conservative_bw=float(peer["peer_effective_GBps"])*1e9
    return dict(
        h2d_median_ms=float(statistics.median(h2d_medians)),
        h2d_p90_ms=float(statistics.median(h2d_p90)),
        p2p_pair_ms=pair,
        gemm_ms_per_row=gemm_slope,
        expert_ms_per_row=expert_slope,
        conservative_peer_bytes_per_s=conservative_bw,
        d2d_sensitivity_ms=[0.02,0.05,0.10,0.20],
        source_files=[
            "hot_expert_replication_threshold_20261003/break_even_microcost.json",
            "timing_stability_numa_20261004/S2B_H2D.json",
            "rank_demand_oracle_20261002/gpu_phase_summary.json",
            "rank_demand_oracle_20261002/rank_load_summary.json",
            "fetch_comm_pareto_p2p_20261002/transport_calibration.csv",
        ],
    )

def pair_cost_ms(n,table):
    if n<=0: return 0.0
    xs=sorted(table)
    if n<=xs[0]: return table[xs[0]]
    for a,b in zip(xs[:-1],xs[1:]):
        if a<=n<=b:
            f=(n-a)/(b-a)
            return table[a]*(1-f)+table[b]*f
    a,b=xs[-2],xs[-1]
    slope=max(0.0,(table[b]-table[a])/(b-a))
    return table[b]+slope*(n-b)

def summarize(result,scope,horizon,cal):
    r=as_result_dict(result);sl=scope_slice(scope,horizon)
    max_rows=r["max_rows"][sl]
    h2d=r["h2d_counts"][sl]
    rep=r["replica_created"][sl]
    srcmiss=r["replica_source_miss"][sl]
    peer_bytes=(r["peer_dispatch_pairs"][sl].sum()+r["peer_return_routes"][sl].sum())*ROW_BYTES
    raw=r["raw_routes"][sl].sum()
    hits=r["global_hit_routes"][sl].sum()
    out=dict(
        events=int(len(max_rows)),
        critical_rows_sum=int(max_rows.sum()),
        critical_rows_peak=int(max_rows.max()) if len(max_rows) else 0,
        mean_rank_cv=float(r["rank_cv"][sl].mean()) if len(max_rows) else 0.0,
        H2D_bytes=int(h2d.sum())*EXPERT_BYTES,
        D2D_bytes=int(rep.sum())*EXPERT_BYTES,
        peer_bytes=int(peer_bytes),
        dispatch_bytes=int(r["peer_dispatch_pairs"][sl].sum())*ROW_BYTES,
        return_bytes=int(r["peer_return_routes"][sl].sum())*ROW_BYTES,
        exact_global_hit_rate=float(hits/raw) if raw else 0.0,
        unique_miss_experts=int(r["unique_misses"][sl].sum()),
        first_fetches=int(r["first_fetches"][sl].sum()),
        reload_fetches=int(r["reload_fetches"][sl].sum()),
        replica_creations=int(rep.sum()),
        replica_from_current_miss=int(srcmiss.sum()),
        replica_served_rows=int(r["replica_served_rows"][sl].sum()),
        replica_unique_victims=int(r["victim_unique"][sl].sum()),
        replica_duplicate_victims=int(r["victim_duplicate"][sl].sum()),
        duplicate_copies_mean=float(r["duplicate_copies"][sl].mean()) if len(max_rows) else 0.0,
        duplicate_copies_peak=int(r["duplicate_copies"][sl].max()) if len(max_rows) else 0,
        evictions=int(r["evictions"][sl].sum()),
        active_peer_pairs_sum=int(r["active_peer_pairs"][sl].sum()),
    )

    fast_comm=0.0;cons_comm=0.0
    alpha=pair_cost_ms(1,cal["p2p_pair_ms"])/2
    bw=cal["conservative_peer_bytes_per_s"]
    idx=np.arange(len(r["max_rows"]))[sl]
    for ev in idx:
        d=int(r["max_dispatch_pair_rows"][ev]);ret=int(r["max_return_pair_rows"][ev])
        fast_comm += pair_cost_ms(d,cal["p2p_pair_ms"])/2 + pair_cost_ms(ret,cal["p2p_pair_ms"])/2
        cons_comm += (alpha + d*ROW_BYTES/bw*1e3 if d else 0.0)
        cons_comm += (alpha + ret*ROW_BYTES/bw*1e3 if ret else 0.0)
    out["comm_fast_ms"]=float(fast_comm)
    out["comm_conservative_ms"]=float(cons_comm)
    out["compute_gemm_ms"]=float(out["critical_rows_sum"]*cal["gemm_ms_per_row"])
    out["compute_expert_ms"]=float(out["critical_rows_sum"]*cal["expert_ms_per_row"])

    transfer={}
    for hname,hms in (("median",cal["h2d_median_ms"]),("p90",cal["h2d_p90_ms"])):
        for d2d in cal["d2d_sensitivity_ms"]:
            overlap=0.0;serial=0.0
            for ev in idx:
                max_fetch=int(r["h2d_counts"][ev].max())
                base=max_fetch*hms
                if r["replica_created"][ev]:
                    dep=hms if r["replica_source_miss"][ev] else 0.0
                    overlap += max(base,dep+d2d)
                    serial += base+d2d
                else:
                    overlap += base;serial += base
            transfer[f"{hname}_d2d{d2d:.2f}_overlap_ms"]=float(overlap)
            transfer[f"{hname}_d2d{d2d:.2f}_serial_ms"]=float(serial)
    out["transfer_models"]=transfer
    t=transfer["median_d2d0.05_overlap_ms"]
    out["phase_fast_gemm_ms"]=float(fast_comm+t+out["compute_gemm_ms"])
    out["phase_fast_expert_ms"]=float(fast_comm+t+out["compute_expert_ms"])
    out["phase_conservative_expert_ms"]=float(cons_comm+t+out["compute_expert_ms"])
    return out

def state_hash(result):
    h=hashlib.sha256()
    r=as_result_dict(result)
    for key in ("slots","owner","primary"):
        h.update(np.ascontiguousarray(r[key]).tobytes())
    return h.hexdigest()

def capacities():
    total=LAYERS*EXPERTS*CACHE_PERCENT//100
    return np.array([total//WORLD+(r<total%WORLD) for r in range(WORLD)],np.int64)

def make_arrays(pool,batch,sample_seed,dp_seed):
    order=pool.candidate(WORLD,batch,sample_seed,dp_seed)
    arrays=pool.pack(order,WORLD,batch)
    return order,arrays

def replay_arrays(arrays,batch,placement_seed,policy,
                  replica=False,hot_mult=1.0,min_gain=0.02,dup_budget=0.05,
                  horizon=HORIZON,replica_all_phase=False):
    return replay_policy(
        arrays["selected"],arrays["offsets"],arrays["prefill_origins"],
        arrays["decode_origins"],arrays["gates"],capacities(),horizon,
        policy,placement_seed,replica,float(hot_mult),float(min_gain),
        float(dup_budget),batch,bool(replica_all_phase))

def run_replay(pool,batch,sample_seed,dp_seed,placement_seed,policy,
               replica=False,hot_mult=1.0,min_gain=0.02,dup_budget=0.05,
               horizon=HORIZON,replica_all_phase=False):
    order,arrays=make_arrays(pool,batch,sample_seed,dp_seed)
    result=replay_arrays(arrays,batch,placement_seed,policy,replica,hot_mult,
                         min_gain,dup_budget,horizon,replica_all_phase)
    return order,arrays,result

def sample_counts(routes,sample):
    out=np.zeros((len(routes),EXPERTS),np.int32)
    for i in range(len(routes)):
        out[i]=np.bincount(np.asarray(routes[i,sample]).reshape(-1),minlength=EXPERTS)
    return out

def hotness_score(counts):
    vals=[]
    for row in counts:
        active=row[row>0]
        if not len(active): continue
        quota=(len(active)+WORLD-1)//WORLD
        vals.append(np.sort(active)[-quota:].sum()/(row.sum()/WORLD))
    return float(np.mean(vals))

@njit(cache=True)
def demand_for_proxy(routes,sample,order,batch):
    ne=len(routes)
    demand=np.zeros((ne,EXPERTS,WORLD),np.int32)
    for ev in range(ne):
        for pos in range(len(sample)):
            req=sample[order[pos]]
            rank=pos//batch
            for k in range(TOPK):
                demand[ev,routes[ev,req,k],rank]+=1
    return demand

@njit(cache=True)
def proxy_placement_scores(demand,seed):
    np.random.seed(seed)
    load_sum=0;load_peak=0;remote_routes=0
    for ev in range(len(demand)):
        totals=demand[ev].sum(1);active=np.flatnonzero(totals>0);m=len(active)
        slots=np.empty(m,np.int8);at=0
        for r in range(WORLD):
            for _ in range(m//WORLD+(r<m%WORLD)):
                slots[at]=r;at+=1
        perm=np.arange(m);np.random.shuffle(perm);assigned=np.empty(m,np.int8)
        for i in range(m):assigned[perm[i]]=slots[i]
        loads=np.zeros(WORLD,np.int64)
        for i in range(m):
            e=active[i];dst=int(assigned[i]);loads[dst]+=totals[e];remote_routes+=totals[e]-demand[ev,e,dst]
        mx=loads.max();load_sum+=int(mx)
        if mx>load_peak:load_peak=int(mx)
    return load_sum,load_peak,int(remote_routes)

def candidate_proxies(pool,batch,dataset):
    routes=np.asarray(pool.a["proxy_routes"])
    n=WORLD*batch
    if batch==256:
        sample_seeds=[0]
    else:
        hot_scored=[];comm_scored=[]
        probe_order=np.random.default_rng(42).permutation(n)
        probe_placement=(7,19,42,73,99,131,172,251)
        for s in range(256):
            sample=np.random.default_rng(s).permutation(pool.n)[:n]
            counts=sample_counts(routes,sample)
            hot_scored.append((hotness_score(counts),s))
            d=demand_for_proxy(routes,sample,probe_order,batch)
            best_remote=0
            for ps in probe_placement:
                _,_,rr=proxy_placement_scores(d,ps)
                if rr>best_remote:best_remote=rr
            comm_scored.append((best_remote,s))
        hot_scored.sort(reverse=True);comm_scored.sort(reverse=True)
        sample_seeds=[]
        for _,s in hot_scored[:16]+comm_scored[:16]:
            if s not in sample_seeds:sample_seeds.append(s)

    candidates=[]
    probe_dp=[0,7,19,42,73,99,131,181,200,251] if batch==128 else list(range(256))
    for s in sample_seeds:
        sample=np.random.default_rng(s).permutation(pool.n)[:n]
        dp_rows=[]
        for dp in probe_dp:
            order=np.random.default_rng(dp).permutation(n)
            d=demand_for_proxy(routes,sample,order,batch)
            div=0
            for ev in range(len(d)):
                div += int(d[ev].max(1).sum())
            dp_rows.append((div,dp,d))
        dp_rows.sort(reverse=True,key=lambda x:(x[0],-x[1]))
        if batch==128 and s in sample_seeds[:2]:
            dp_rows=[]
            for dp in range(256):
                order=np.random.default_rng(dp).permutation(n)
                d=demand_for_proxy(routes,sample,order,batch)
                div=sum(int(d[ev].max(1).sum()) for ev in range(len(d)))
                dp_rows.append((div,dp,d))
            dp_rows.sort(reverse=True,key=lambda x:(x[0],-x[1]))
        for div,dp,d in dp_rows[:4]:
            comm=[];load=[]
            for ps in range(256):
                ls,lp,rr=proxy_placement_scores(d,ps)
                comm.append((rr,ls,lp,ps))
                load.append((ls,lp,rr,ps))
            comm.sort(reverse=True)
            load.sort(reverse=True)
            for family,rows in (("COMM",comm[:2]),("LOAD",load[:2])):
                for vals in rows:
                    if family=="COMM":
                        rr,ls,lp,ps=vals
                    else:
                        ls,lp,rr,ps=vals
                    candidates.append(dict(dataset=dataset,batch=batch,family=family,
                                           sample_seed=s,dp_seed=dp,placement_seed=ps,
                                           proxy_remote_routes=int(rr),
                                           proxy_critical_rows=int(ls),
                                           proxy_peak_rows=int(lp),
                                           dp_divergence=int(div)))
    out=[]
    for fam in ("COMM","LOAD"):
        rows=[x for x in candidates if x["family"]==fam]
        key=(lambda x:(-x["proxy_remote_routes"],-x["proxy_critical_rows"],x["sample_seed"],x["dp_seed"],x["placement_seed"])) if fam=="COMM" else (
             lambda x:(-x["proxy_critical_rows"],-x["proxy_peak_rows"],-x["proxy_remote_routes"],x["sample_seed"],x["dp_seed"],x["placement_seed"]))
        rows.sort(key=key)
        seen=set()
        for row in rows:
            ident=(row["sample_seed"],row["dp_seed"],row["placement_seed"])
            if ident in seen: continue
            seen.add(ident);out.append(row)
            if len(seen)>=12: break
    return out

def exact_seed_search(cal):
    all_exact=[]
    for batch in (128,256):
        for dataset in ("MATH","ShareGPT"):
            pool=Pool(dataset)
            cand=candidate_proxies(pool,batch,dataset)
            for i,row in enumerate(cand):
                _,arrays,result=run_replay(pool,batch,row["sample_seed"],row["dp_seed"],
                                           row["placement_seed"],0,False,horizon=HORIZON)
                dec=summarize(result,"decode",HORIZON,cal)
                pre=summarize(result,"prefill",HORIZON,cal)
                exact=dict(**row,decode=dec,prefill=pre,state_sha256=state_hash(result))
                all_exact.append(exact)
                print(json.dumps(dict(stage="seed-exact",batch=batch,dataset=dataset,
                                      family=row["family"],done=i+1,total=len(cand),
                                      peer=dec["peer_bytes"],critical=dec["critical_rows_sum"])),flush=True)

    winners={}
    top5={}
    for batch in (128,256):
        for fam in ("COMM","LOAD"):
            rows=[x for x in all_exact if x["batch"]==batch and x["family"]==fam]
            if fam=="COMM":
                rows.sort(key=lambda x:(-x["decode"]["peer_bytes"],
                                        -x["decode"]["critical_rows_sum"],
                                        x["dataset"],x["sample_seed"],x["dp_seed"],x["placement_seed"]))
            else:
                rows.sort(key=lambda x:(-x["decode"]["critical_rows_sum"],
                                        -x["decode"]["critical_rows_peak"],
                                        -x["decode"]["peer_bytes"],
                                        x["dataset"],x["sample_seed"],x["dp_seed"],x["placement_seed"]))
            winners[(batch,fam)]=rows[0]
            top5[(batch,fam)]=rows[:5]
    write(OUT/"seed_exact.json",dict(rows=all_exact))
    return winners,top5

REPLICA_CONFIGS=[
    dict(hot_mult=h,min_gain=g,dup_budget=b)
    for h,g,b in itertools.product((1.0,1.5,2.0),(0.02,0.05),(0.02,0.05))
]

def policy_eval(winners,cal):
    rows=[]
    for (batch,fam),win in sorted(winners.items()):
        pool=Pool(win["dataset"])
        sample=win["sample_seed"];dp=win["dp_seed"];seed=win["placement_seed"]
        order,arrays=make_arrays(pool,batch,sample,dp)
        baselines={}
        baseline64={}
        for p in (0,1,2):
            res=replay_arrays(arrays,batch,seed,p,False,horizon=HORIZON)
            entry=dict(batch=batch,stress=fam,dataset=win["dataset"],policy=POLICY_NAMES[p],
                       replica=False,config=None,scope="decode",
                       manifest_sha256=digest(dict(request_ids=order.tolist(),sample=sample,dp=dp,seed=seed)),
                       decode=summarize(res,"decode",HORIZON,cal),
                       prefill=summarize(res,"prefill",HORIZON,cal),
                       moe_total=summarize(res,"moe_total",HORIZON,cal),
                       state_sha256=state_hash(res))
            rows.append(entry);baselines[p]=entry
            b64=replay_arrays(arrays,batch,seed,p,False,horizon=64)
            baseline64[p]=summarize(b64,"decode",64,cal)

        for p in (0,1,2):
            short=[]
            base64=baseline64[p]
            for cfg in REPLICA_CONFIGS:
                res=replay_arrays(arrays,batch,seed,p,True,**cfg,horizon=64,
                                  replica_all_phase=False)
                dec=summarize(res,"decode",64,cal)
                short.append(dict(config=cfg,decode=dec))
            by_compute=min(short,key=lambda x:x["decode"]["critical_rows_sum"])
            by_phase=min(short,key=lambda x:x["decode"]["phase_fast_expert_ms"])
            safe=[x for x in short if x["decode"]["H2D_bytes"]<=base64["H2D_bytes"]*1.02]
            by_safe=min(safe,key=lambda x:x["decode"]["phase_fast_expert_ms"]) if safe else by_phase
            selected=[]
            seen=set()
            for x,label in ((by_compute,"compute"),(by_phase,"phase"),(by_safe,"h2d-safe")):
                key=tuple(sorted(x["config"].items()))
                if key not in seen:
                    selected.append((x["config"],[label]));seen.add(key)
                else:
                    for cfg,labels in selected:
                        if tuple(sorted(cfg.items()))==key:labels.append(label)
            full=[]
            for cfg,labels in selected:
                res=replay_arrays(arrays,batch,seed,p,True,**cfg,horizon=HORIZON,
                                  replica_all_phase=False)
                item=dict(batch=batch,stress=fam,dataset=win["dataset"],
                          policy=POLICY_NAMES[p]+"+REP",base_policy=POLICY_NAMES[p],
                          replica=True,config=cfg,selection_labels=labels,scope="decode-only",
                          manifest_sha256=digest(dict(request_ids=order.tolist(),sample=sample,dp=dp,seed=seed)),
                          decode=summarize(res,"decode",HORIZON,cal),
                          prefill=summarize(res,"prefill",HORIZON,cal),
                          moe_total=summarize(res,"moe_total",HORIZON,cal),
                          state_sha256=state_hash(res))
                rows.append(item);full.append(item)
            best=min(full,key=lambda x:x["decode"]["phase_fast_expert_ms"])
            res=replay_arrays(arrays,batch,seed,p,True,**best["config"],
                              horizon=HORIZON,replica_all_phase=True)
            rows.append(dict(batch=batch,stress=fam,dataset=win["dataset"],
                             policy=POLICY_NAMES[p]+"+REP",base_policy=POLICY_NAMES[p],
                             replica=True,config=best["config"],selection_labels=["all-phase-check"],
                             scope="all-phase",
                             manifest_sha256=digest(dict(request_ids=order.tolist(),sample=sample,dp=dp,seed=seed)),
                             decode=summarize(res,"decode",HORIZON,cal),
                             prefill=summarize(res,"prefill",HORIZON,cal),
                             moe_total=summarize(res,"moe_total",HORIZON,cal),
                             state_sha256=state_hash(res)))
    write(OUT/"policy_rows.json",dict(rows=rows))
    return rows

def best_rows(rows):
    out={}
    for batch in (128,256):
        for fam in ("COMM","LOAD"):
            subset=[x for x in rows if x["batch"]==batch and x["stress"]==fam and x["scope"]!="all-phase"]
            base={x["policy"]:x for x in subset if not x["replica"]}
            chosen=[]
            for pname in ("BR","CA","LA"):
                chosen.append(base[pname])
                reps=[x for x in subset if x["policy"]==pname+"+REP"]
                b=base[pname]["decode"]
                safe=[x for x in reps if x["decode"]["H2D_bytes"]<=b["H2D_bytes"]*1.02]
                pool=safe or reps
                chosen.append(min(pool,key=lambda x:x["decode"]["phase_fast_expert_ms"]))
            out[(batch,fam)]=chosen
    return out

def write_summary(winners,top5,rows,cal):
    selected=best_rows(rows)
    table=[]
    selected_full={}
    for key,items in selected.items():
        batch,fam=key
        selected_full[f"B{batch}_{fam}"]=items
        for x in items:
            d=x["decode"]
            table.append(dict(batch=batch,stress=fam,dataset=x["dataset"],policy=x["policy"],
                              scope=x["scope"],config=json.dumps(x["config"],sort_keys=True) if x["config"] else "",
                              critical_rows=d["critical_rows_sum"],H2D_bytes=d["H2D_bytes"],
                              D2D_bytes=d["D2D_bytes"],peer_bytes=d["peer_bytes"],
                              exact_hit=d["exact_global_hit_rate"],
                              phase_fast_expert_ms=d["phase_fast_expert_ms"],
                              phase_cons_expert_ms=d["phase_conservative_expert_ms"],
                              replica_creations=d["replica_creations"]))
    with (PACKET/"comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)

    gain_ratio=[]
    for fam in ("COMM","LOAD"):
        byb={}
        for batch in (128,256):
            items={x["policy"]:x for x in selected[(batch,fam)]}
            def gain(a,b,metric):
                av=items[a]["decode"][metric];bv=items[b]["decode"][metric]
                return (av-bv)/av if av else 0.0
            byb[batch]=dict(
                CA_over_BR_rows=gain("BR","CA","critical_rows_sum"),
                LA_over_BR_rows=gain("BR","LA","critical_rows_sum"),
                BR_REP_over_BR_rows=gain("BR","BR+REP","critical_rows_sum"),
                CA_REP_over_CA_rows=gain("CA","CA+REP","critical_rows_sum"),
                LA_REP_over_LA_rows=gain("LA","LA+REP","critical_rows_sum"),
                CA_over_BR_phase=gain("BR","CA","phase_fast_expert_ms"),
                LA_over_BR_phase=gain("BR","LA","phase_fast_expert_ms"),
                BR_REP_over_BR_phase=gain("BR","BR+REP","phase_fast_expert_ms"),
                CA_REP_over_CA_phase=gain("CA","CA+REP","phase_fast_expert_ms"),
                LA_REP_over_LA_phase=gain("LA","LA+REP","phase_fast_expert_ms"),
            )
        row=dict(stress=fam,B128=byb[128],B256=byb[256],growth={})
        for k in byb[128]:
            a=byb[128][k];b=byb[256][k]
            row["growth"][k]=None if abs(a)<1e-12 else b/a
        gain_ratio.append(row)

    all_phase=[x for x in rows if x["scope"]=="all-phase"]
    win_json={f"B{b}_{fam}":v for (b,fam),v in winners.items()}
    top_json={f"B{b}_{fam}":v for (b,fam),v in top5.items()}
    result=dict(status="PASS",calibration=cal,winners=win_json,top5=top_json,
                selected=table,selected_full=selected_full,
                batch_headroom_growth=gain_ratio,
                all_phase_replica_checks=all_phase)
    write(PACKET/"RESULT.json",result)
    write(PACKET/"prefill_decode_comparison.json",
          dict(selected=selected_full,all_phase_replica_checks=all_phase,
               note="Prefill timings are MoE-phase model only; attention/dense prefill kernels excluded."))

    lines=[
        "# R8 B128/B256 phase-aware policy CPU result","",
        "Stress workloads are intentionally optimized upper/headroom cases, not dataset averages.","",
        "## Stress winners","",
        "| Batch | Stress | Dataset | sample / DP / placement | BR peer GiB | BR critical rows |",
        "|---:|---|---|---|---:|---:|",
    ]
    for (b,fam),v in sorted(winners.items()):
        d=v["decode"];lines.append(
            f"| {b} | {fam} | {v['dataset']} | {v['sample_seed']} / {v['dp_seed']} / {v['placement_seed']} | {d['peer_bytes']/2**30:.2f} | {d['critical_rows_sum']} |")
    lines += ["","## Selected policy comparison","",
              "| B | Stress | Policy | Critical rows | H2D TiB | D2D GiB | Peer GiB | phase fast/expert ms |",
              "|---:|---|---|---:|---:|---:|---:|---:|"]
    for r in table:
        lines.append(f"| {r['batch']} | {r['stress']} | {r['policy']} | {r['critical_rows']} | {r['H2D_bytes']/2**40:.3f} | {r['D2D_bytes']/2**30:.3f} | {r['peer_bytes']/2**30:.3f} | {r['phase_fast_expert_ms']:.1f} |")
    lines += ["","## Batch-headroom growth",""]
    for row in gain_ratio:
        lines.append(f"- {row['stress']}: B128={json.dumps(row['B128'],sort_keys=True)}")
        lines.append(f"  B256={json.dumps(row['B256'],sort_keys=True)}")
        lines.append(f"  B256/B128 gain ratios={json.dumps(row['growth'],sort_keys=True)}")
    lines += ["",
              "CPU/calibrated model only. `phase fast/expert` uses archived fast-P2P microcost, median H2D, D2D=0.05 ms sensitivity point, and the archived enclosing expert-kernel row slope.",
              "See RESULT.json and prefill_decode_comparison.json for prefill/decode separation, serial-vs-overlap D2D transfer models, p90 H2D and D2D sensitivity."]
    (PACKET/"RESULTS.md").write_text("\n".join(lines)+"\n")

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cal=load_calibration()
    write(PACKET/"calibration.json",cal)
    started=time.time()
    winners,top5=exact_seed_search(cal)
    write(PACKET/"stress_winners.json",{f"B{b}_{fam}":v for (b,fam),v in winners.items()})
    rows=policy_eval(winners,cal)
    write_summary(winners,top5,rows,cal)
    write(PACKET/"status.json",dict(status="PASS",seconds=time.time()-started,
                                    cpu_only=True,physical_timing=False))
    print(json.dumps(dict(status="PASS",seconds=time.time()-started),indent=2))

if __name__=="__main__":
    main()
