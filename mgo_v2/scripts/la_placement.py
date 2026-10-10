"""Side-effect-free LA placement for up to eight ranks, matching the calibrated CPU oracle."""
import numpy as np
from numba import njit
WORLD=8
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
    experts=demand.shape[0]
    base=layer*experts
    for ii in range(experts):
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



@njit(cache=True)
def load_locality_near_assignment(demand, misses, owner, layer, near_bps=200, quota_override=None):
    """Low-cost LA-first, communication-second admission.

    1) Preserve the balanced miss-count quota used by BR/LA.
    2) Place heavy misses first.
    3) Compute the minimum projected critical-rank expert-row load.
    4) Admit any destination within near_bps (default 2%) of that optimum.
    5) Inside that near-tie set, maximize rank-local demand, which minimizes
       remote expert-route traffic without scanning token-level packet masks.
    6) Tie-break by lower projected critical load, destination load, rank id.

    Complexity is O(num_misses * world^2), with world <= 8.  Unlike packet
    fanout CA, it never scans all tokens for every candidate destination.
    """
    world=demand.shape[1]
    m=len(misses)
    quota=np.empty(world,np.int64)
    for r in range(world):
        quota[r]=m//world+(r<m%world) if quota_override is None else quota_override[r]
    assert quota.sum()==m and quota.max()-quota.min()<=1
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
        projected=np.full(world,2**60,np.int64)
        best_comp=2**60
        for r in range(world):
            if remaining[r]<=0:
                continue
            mx=0
            for rr in range(world):
                v=loads[rr]+(total if rr==r else 0)
                if v>mx:mx=v
            projected[r]=mx
            if mx<best_comp:best_comp=mx
        slack=max(1,(best_comp*near_bps)//10000)
        limit=best_comp+slack
        best=-1
        best_local=-1
        best_comp2=2**60
        best_dst=2**60
        for r in range(world):
            if remaining[r]<=0 or projected[r]>limit:
                continue
            local=demand[e,r]
            dst=loads[r]+total
            if (local>best_local or
                (local==best_local and projected[r]<best_comp2) or
                (local==best_local and projected[r]==best_comp2 and dst<best_dst) or
                (local==best_local and projected[r]==best_comp2 and dst==best_dst and (best<0 or r<best))):
                best=r;best_local=local;best_comp2=projected[r];best_dst=dst
        assert best>=0
        answer[i]=best
        remaining[best]-=1
        loads[best]+=total
    assert remaining.sum()==0
    return answer
