"""Single-thread compiled frozen-route CPU replay; no Torch/CUDA imports."""
import numpy as np
from numba import njit

ROW_BYTES=4096
EXPERT_BYTES=9437184
METRICS=['raw_routes','raw_gate_mass','exact_global_hits','exact_global_hit_mass',
 'exact_local_hits','exact_local_hit_mass','resident_substitute_hits','resident_substitute_mass',
 'shared_admission_substitutions','shared_admission_mass','substituted_routes','substituted_mass',
 'effective_hits','effective_hit_mass','residual_miss_routes','residual_miss_mass',
 'effective_local_hits','effective_local_hit_mass','unique_raw_expert_events','unique_miss_expert_events',
 'first_fetches','reload_fetches','replica_fetches','evictions','resident_copies','unique_resident_experts',
 'duplicate_copies','local_services','effective_routes','remote_expert_routes','remote_token_rank_pairs',
 'local_service_gate_mass','effective_gate_mass','replica_victim_reloads','replicas_reused',
 'replicas_evicted','replicas_evicted_without_reuse','replica_survival_events_sum',
 'replica_score_lt_quarter','replica_score_quarter_half','replica_score_half_one',
 'replica_score_one_two','replica_score_ge_two','replica_blocked_by_pinning','max_rank_mandatory_fetches',
 'min_rank_mandatory_fetches','protected_miss_experts','substituted_experts']

@njit(cache=True)
def balanced_assignment(demand,experts,world,randomized,quota=None):
    """Exact Hungarian local-route objective; BR shuffles the same quota slots."""
    n=len(experts);slots=np.empty(n,np.int64);at=0
    for r in range(world):
        count=n//world+(r<n%world) if quota is None else quota[r]
        assert count>=0
        for _ in range(count):
            assert at<n
            slots[at]=r;at+=1
    assert at==n
    if randomized:
        order=np.arange(n);np.random.shuffle(order);answer=np.empty(n,np.int64)
        for j in range(n):answer[order[j]]=slots[j]
        return answer
    u=np.zeros(n+1,np.int64);v=np.zeros(n+1,np.int64);p=np.zeros(n+1,np.int64);way=np.zeros(n+1,np.int64)
    for i in range(1,n+1):
        p[0]=i;j0=0;minimum=np.full(n+1,2**50,np.int64);used=np.zeros(n+1,np.bool_)
        while True:
            used[j0]=True;i0=p[j0];delta=2**50;j1=0
            for j in range(1,n+1):
                if not used[j]:
                    value=-demand[experts[i0-1],slots[j-1]]-u[i0]-v[j]
                    if value<minimum[j]:minimum[j]=value;way[j]=j0
                    if minimum[j]<delta:delta=minimum[j];j1=j
            for j in range(n+1):
                if used[j]:u[p[j]]+=delta;v[j]-=delta
                else:minimum[j]-=delta
            j0=j1
            if p[j0]==0:break
        while True:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if j0==0:break
    answer=np.empty(n,np.int64)
    for j in range(1,n+1):answer[p[j]-1]=slots[j-1]
    return answer

@njit(cache=True)
def choose_slot(rank,layer,active,slot_keys,capacities,last_used,gates,gate_eviction,experts):
    best=-1;best_gate=np.inf;best_used=2**60;best_key=2**60
    for slot in range(capacities[rank]):
        key=slot_keys[rank,slot]
        if key<0:return slot
        if key//experts==layer and active[key%experts]:continue
        score=gates[key//experts,key%experts] if gate_eviction else 0.
        used=last_used[rank,key]
        if score<best_gate or (score==best_gate and (used<best_used or (used==best_used and key<best_key))):
            best=slot;best_gate=score;best_used=used;best_key=key
    return best

@njit(cache=True)
def place(rank,key,slot,is_replica,tick,slot_keys,owner,primary,last_used,seen,lost_replica,birth,reuses,row):
    victim=slot_keys[rank,slot]
    if victim>=0:
        row[23]+=1
        if birth[rank,victim]>=0:
            row[35]+=1;row[36]+=reuses[rank,victim]==0;row[37]+=tick-birth[rank,victim]
            birth[rank,victim]=-1;reuses[rank,victim]=0
        owner[victim]&=~(1<<rank)
        if owner[victim]==0:primary[victim]=-1;lost_replica[victim]=is_replica
        elif primary[victim]==rank:
            for r in range(len(slot_keys)):
                if owner[victim]&(1<<r):primary[victim]=r;break
    assert not owner[key]&(1<<rank)
    if owner[key]==0:
        assert not is_replica
        primary[key]=rank
        if seen[key]:row[21]+=1;row[33]+=lost_replica[key]
        else:row[20]+=1
    else:
        assert is_replica
        row[22]+=1;birth[rank,key]=tick;reuses[rank,key]=0
    owner[key]|=1<<rank;slot_keys[rank,slot]=key;last_used[rank,key]=tick;seen[key]=True;lost_replica[key]=False

@njit(cache=True)
def suffix_demand(selected,offsets,origins,layers,experts,world,horizon):
    """Strictly later raw decode routes; no future substitution/cache state."""
    out=np.zeros((horizon+1,layers,experts,world),np.int32)
    running=np.zeros((layers,experts,world),np.int32)
    for step in range(horizon,0,-1):
        out[step]=running
        for layer in range(layers):
            event=step*layers+layer;lo=offsets[event]
            for token in range(len(origins)):
                r=origins[token]
                for k in range(selected.shape[1]):running[layer,selected[lo+token,k],r]+=1
    out[0]=running
    return out

@njit(cache=True)
def replay(selected,weights,offsets,origins0,origins,gates_by_event,similarity,capacities,horizon,gate_eviction,substitution,policy,seed):
    np.random.seed(seed)
    layers,experts=similarity.shape[:2];world=len(capacities);keys=layers*experts;topk=selected.shape[1];events=(horizon+1)*layers
    slots=np.full((world,int(capacities.max())),-1,np.int32);owner=np.zeros(keys,np.int16);primary=np.full(keys,-1,np.int8)
    last=np.zeros((world,keys),np.int32);seen=np.zeros(keys,np.bool_);lost=np.zeros(keys,np.bool_)
    birth=np.full((world,keys),-1,np.int32);reuses=np.zeros((world,keys),np.int32);gates=np.zeros((layers,experts),np.float32)
    future=suffix_demand(selected,offsets,origins,layers,experts,world,horizon) if policy==2 else np.zeros((1,1,1,1),np.int32)
    rows=np.zeros((events,48),np.float64);rank_fetches=np.zeros((events,world),np.int32)
    for event in range(events):
        layer=event%layers;step=event//layers;tick=event+1;row=rows[event];gates[layer]=gates_by_event[event]
        org=origins0 if step==0 else origins;lo=offsets[event];hi=offsets[event+1];n=hi-lo;assert n==len(org)
        raw_active=np.zeros(experts,np.bool_);maxweight=np.zeros(experts,np.float32)
        for t in range(n):
            for k in range(topk):
                e=selected[lo+t,k];raw_active[e]=True;maxweight[e]=max(maxweight[e],weights[lo+t,k])
        resident=owner[layer*experts:(layer+1)*experts]!=0
        protected=raw_active&(~resident)&(maxweight>=.20);tier1=raw_active&resident|protected
        targets=np.arange(experts);mapped=np.zeros(experts,np.bool_)
        if substitution:
            for e in range(experts):
                if raw_active[e] and not resident[e] and not protected[e]:
                    best=-1;score=-1.
                    for j in range(experts):
                        value=similarity[layer,e,j]
                        if j!=e and tier1[j] and value>=np.float32(.65) and value>score:best=j;score=value
                    if best<0:
                        for j in range(experts):
                            value=similarity[layer,e,j]
                            if j!=e and resident[j] and value>=np.float32(.65) and value>score:best=j;score=value
                    if best>=0:targets[e]=best;mapped[e]=True
        # Pre-admission, disjoint source-route accounting. Shared new anchors
        # remain explicit rather than being mislabeled resident cache hits.
        effective=np.full((n,topk),-1,np.int16);masses=np.zeros((n,topk),np.float64);lengths=np.zeros(n,np.int8);demand=np.zeros((experts,world),np.int64);active=np.zeros(experts,np.bool_)
        for t in range(n):
            r=org[t]
            for k in range(topk):
                e=selected[lo+t,k];w=weights[lo+t,k];target=targets[e];key=layer*experts+e;targetkey=layer*experts+target
                row[0]+=1;row[1]+=w
                if resident[e]:
                    row[2]+=1;row[3]+=w;row[12]+=1;row[13]+=w
                    if owner[key]&(1<<r):row[4]+=1;row[5]+=w
                elif mapped[e]:
                    row[10]+=1;row[11]+=w
                    if resident[target]:row[6]+=1;row[7]+=w;row[12]+=1;row[13]+=w
                    else:row[8]+=1;row[9]+=w
                else:row[14]+=1;row[15]+=w
                if owner[targetkey]&(1<<r):row[16]+=1;row[17]+=w
                position=-1
                for j in range(lengths[t]):
                    if effective[t,j]==target:position=j;break
                if position<0:
                    position=lengths[t];lengths[t]+=1;effective[t,position]=target;demand[target,r]+=1;active[target]=True
                masses[t,position]+=w
        row[18]=raw_active.sum();row[19]=(active&(~resident)).sum();row[46]=protected.sum();row[47]=mapped.sum()
        misses=np.flatnonzero(active&(~resident));assignment=balanced_assignment(demand,misses,world,policy==0)
        for i in range(len(misses)):rank_fetches[event,assignment[i]]+=1
        row[44]=rank_fetches[event].max();row[45]=rank_fetches[event].min();assert row[44]-row[45]<=1
        for i in range(len(misses)):
            e=misses[i];r=assignment[i];key=layer*experts+e
            slot=choose_slot(r,layer,active,slots,capacities,last,gates,gate_eviction,experts);assert slot>=0
            place(r,key,slot,False,tick,slots,owner,primary,last,seen,lost,birth,reuses,row)
        if policy==2:
            for i in range(len(misses)):
                e=misses[i];key=layer*experts+e;best=-1;count=-1
                for r in range(world):
                    if r!=primary[key] and future[step,layer,e,r]>count:best=r;count=future[step,layer,e,r]
                value=count*2*ROW_BYTES
                bucket=0 if value<EXPERT_BYTES/4 else 1 if value<EXPERT_BYTES/2 else 2 if value<EXPERT_BYTES else 3 if value<2*EXPERT_BYTES else 4
                row[38+bucket]+=1
                if value>=EXPERT_BYTES:
                    slot=choose_slot(best,layer,active,slots,capacities,last,gates,gate_eviction,experts)
                    if slot<0:row[43]+=1
                    else:place(best,key,slot,True,tick,slots,owner,primary,last,seen,lost,birth,reuses,row)
        served=np.zeros((world,experts),np.bool_)
        for t in range(n):
            r=org[t];dispatch=0
            for j in range(lengths[t]):
                e=effective[t,j];key=layer*experts+e;dst=r if owner[key]&(1<<r) else primary[key];assert dst>=0 and owner[key]&(1<<dst)
                row[28]+=1;row[32]+=masses[t,j];served[dst,e]=True
                if dst==r:row[27]+=1;row[31]+=masses[t,j]
                else:row[29]+=1;dispatch|=1<<dst
            for r2 in range(world):row[30]+=(dispatch>>r2)&1
        for r in range(world):
            for e in range(experts):
                if served[r,e]:
                    key=layer*experts+e
                    if birth[r,key]>=0 and birth[r,key]<tick:
                        if reuses[r,key]==0:row[34]+=1
                        reuses[r,key]+=1
                    last[r,key]=tick
        row[24]=np.count_nonzero(slots>=0);row[25]=np.count_nonzero(owner);row[26]=row[24]-row[25]
        assert row[2]+row[6]+row[8]+row[14]==row[0] and row[27]+row[29]==row[28]
    reconstructed=np.zeros(keys,np.int16)
    for r in range(world):
        for slot in range(capacities[r]):
            key=slots[r,slot]
            if key>=0:assert not reconstructed[key]&(1<<r);reconstructed[key]|=1<<r
    assert np.array_equal(reconstructed,owner)
    alive_replicas=np.count_nonzero(birth>=0);alive_unused=np.count_nonzero((birth>=0)&(reuses==0))
    return rows,rank_fetches,slots,owner,primary,last,seen,lost,birth,reuses,np.array([alive_replicas,alive_unused],np.int64)
