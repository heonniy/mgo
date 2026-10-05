"""Online packet-aware admission for the fused token/rank MoE transport.

The communication unit is a remote token->destination-rank packet, not an
expert route. Multiple experts of one token placed on the same destination
rank share one forward packet and one coalesced return packet.

All policies preserve the same balanced miss-count H2D quotas. FCA minimizes
a rank-incident packet makespan proxy and then total remote packet count.
LA+CA is intentionally lexicographic (compute first, communication second) so
it does not introduce an uncalibrated lambda between rows and packets.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _balanced_quotas(n, world):
    out=np.empty(world,np.int64)
    for r in range(world):
        out[r]=n//world+(r<n%world)
    return out


@njit(cache=True)
def _prepare_packet_state(effective,lengths,origins,primary,layer,experts,misses,world):
    n=len(origins);m=len(misses);base=layer*experts
    position=np.full(experts,-1,np.int64)
    for i in range(m):position[misses[i]]=i
    presence=np.zeros((m,n),np.bool_)
    masks=np.zeros(n,np.int64)
    for t in range(n):
        for j in range(lengths[t]):
            e=int(effective[t,j]);i=position[e]
            if i>=0:
                presence[i,t]=True
            else:
                r=int(primary[base+e])
                if r>=0:masks[t]|=1<<r
    incident=np.zeros(world,np.int64)
    for t in range(n):
        src=int(origins[t]);mask=masks[t]
        for r in range(world):
            if r!=src and (mask&(1<<r)):
                incident[src]+=1;incident[r]+=1
    return presence,masks,incident


@njit(cache=True)
def _candidate_packet_cost(i,rank,presence,masks,origins,incident,world):
    src_delta=np.zeros(world,np.int64);delta=0
    for t in range(len(origins)):
        if not presence[i,t]:continue
        src=int(origins[t])
        if src==rank or (masks[t]&(1<<rank)):continue
        delta+=1;src_delta[src]+=1
    mx=0
    for r in range(world):
        v=incident[r]+src_delta[r]+(delta if r==rank else 0)
        if v>mx:mx=v
    return mx,delta


@njit(cache=True)
def _commit_packet_choice(i,rank,presence,masks,origins,incident):
    delta=0
    for t in range(len(origins)):
        if not presence[i,t]:continue
        src=int(origins[t])
        if src==rank or (masks[t]&(1<<rank)):continue
        masks[t]|=1<<rank
        incident[src]+=1;incident[rank]+=1;delta+=1
    return delta


@njit(cache=True)
def _impact_order(presence,misses):
    m=len(misses);impact=np.zeros(m,np.int64)
    for i in range(m):
        c=0
        for t in range(presence.shape[1]):c+=presence[i,t]
        impact[i]=c*1024-misses[i]
    return np.argsort(-impact)


@njit(cache=True)
def fanout_assignment(effective,lengths,origins,primary,layer,experts,misses,world):
    """Balanced-quota dynamic fanout-aware CA.

    Greedy objective:
      1) minimum max rank-incident remote token/rank packets,
      2) minimum newly-created remote token/rank packets,
      3) lower rank id.

    Every committed placement updates token destination masks before the next
    expert is evaluated, capturing packet coalescing that a static expert/rank
    cost matrix cannot represent.
    """
    m=len(misses);quota=_balanced_quotas(m,world);remaining=quota.copy()
    answer=np.empty(m,np.int64)
    if m==0:return answer
    presence,masks,incident=_prepare_packet_state(
        effective,lengths,origins,primary,layer,experts,misses,world)
    order=_impact_order(presence,misses)
    for oi in range(m):
        i=int(order[oi]);best=-1;best_max=2**60;best_delta=2**60
        for r in range(world):
            if remaining[r]<=0:continue
            mx,delta=_candidate_packet_cost(i,r,presence,masks,origins,incident,world)
            if (mx<best_max or
                (mx==best_max and delta<best_delta) or
                (mx==best_max and delta==best_delta and (best<0 or r<best))):
                best=r;best_max=mx;best_delta=delta
        assert best>=0
        answer[i]=best;remaining[best]-=1
        _commit_packet_choice(i,best,presence,masks,origins,incident)
    assert remaining.sum()==0
    return answer


@njit(cache=True)
def _first_owner(mask,world):
    for r in range(world):
        if mask&(1<<r):return r
    return -1


@njit(cache=True)
def _estimate_existing_loads(demand,owner,layer,experts,world):
    """Single-copy mainline load estimate; replication is outside this method."""
    loads=np.zeros(world,np.int64);base=layer*experts
    for e in range(experts):
        total=0
        for r in range(world):total+=demand[e,r]
        if total==0:continue
        dst=_first_owner(int(owner[base+e]),world)
        if dst>=0:loads[dst]+=total
    return loads


@njit(cache=True)
def load_fanout_assignment(demand,effective,lengths,origins,owner,primary,layer,experts,misses,world):
    """Balanced-quota LA+CA with no uncalibrated cross-unit weight.

    Lexicographic objective:
      1) minimum critical-rank expert rows,
      2) minimum max rank-incident token/rank packets,
      3) minimum new remote packet count,
      4) minimum destination expert load,
      5) maximum local expert demand,
      6) lower rank id.
    """
    m=len(misses);quota=_balanced_quotas(m,world);remaining=quota.copy()
    answer=np.empty(m,np.int64)
    if m==0:return answer
    presence,masks,incident=_prepare_packet_state(
        effective,lengths,origins,primary,layer,experts,misses,world)
    loads=_estimate_existing_loads(demand,owner,layer,experts,world)
    totals=np.zeros(m,np.int64)
    for i in range(m):
        for r in range(world):totals[i]+=demand[misses[i],r]
    order=np.argsort(-totals)
    for oi in range(m):
        i=int(order[oi]);e=int(misses[i]);total=int(totals[i])
        best=-1;best_comp=2**60;best_comm=2**60;best_delta=2**60
        best_dst=2**60;best_local=-1
        for r in range(world):
            if remaining[r]<=0:continue
            comp=0
            for rr in range(world):
                v=loads[rr]+(total if rr==r else 0)
                if v>comp:comp=v
            comm,delta=_candidate_packet_cost(i,r,presence,masks,origins,incident,world)
            dst=loads[r]+total;local=int(demand[e,r])
            if (comp<best_comp or
                (comp==best_comp and comm<best_comm) or
                (comp==best_comp and comm==best_comm and delta<best_delta) or
                (comp==best_comp and comm==best_comm and delta==best_delta and dst<best_dst) or
                (comp==best_comp and comm==best_comm and delta==best_delta and dst==best_dst and local>best_local) or
                (comp==best_comp and comm==best_comm and delta==best_delta and dst==best_dst and local==best_local and (best<0 or r<best))):
                best=r;best_comp=comp;best_comm=comm;best_delta=delta
                best_dst=dst;best_local=local
        assert best>=0
        answer[i]=best;remaining[best]-=1;loads[best]+=total
        _commit_packet_choice(i,best,presence,masks,origins,incident)
    assert remaining.sum()==0
    return answer
