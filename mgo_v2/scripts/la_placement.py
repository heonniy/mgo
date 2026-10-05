"""Side-effect-free LA placement for up to eight ranks, matching the calibrated CPU oracle."""
import numpy as np
from numba import njit
WORLD=8
EXPERTS=128
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
    loads=np.zeros(demand.shape[1],np.int64)
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
    world=demand.shape[1]
    m=len(misses)
    quota=np.empty(world,np.int64)
    for r in range(world):
        quota[r]=m//world+(r<m%world)
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
        for r in range(world):
            if remaining[r]<=0:
                continue
            mx=0
            for rr in range(world):
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

