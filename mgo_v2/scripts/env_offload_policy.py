"""Live one-event adapter of the frozen CPU policy, used only during PLAN."""
import numpy as np
from numba import njit
from br_carep_cpu import balanced_assignment,choose_slot,place,ROW_BYTES,EXPERT_BYTES
from old_ca_fanout_policy import fanout_assignment
from la_placement import load_assignment,load_locality_near_assignment
from mgo_v2.fanout_admission import fanout_assignment as packet_fanout_assignment, load_fanout_assignment
@njit(cache=True)
def seed_rng(seed):np.random.seed(seed)
@njit(cache=True)
def step(event,selected,weights,origins,gate_scores,similarity,capacities,substitution,policy,future,slots,owner,primary,last,seen,lost,birth,reuses,gates):
    layers,experts=similarity.shape[:2];world=len(capacities);gate_eviction=True
    rank_fetches=np.zeros(world,np.int32)
    fetches=[(0,0,0,0,0)];fetches.pop()
    layer=event%layers;step=event//layers;tick=event+1;row=np.zeros(48,np.float64);gates[layer]=gate_scores
    org=origins;lo=0;n=len(org);topk=selected.shape[1]
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
    misses=np.flatnonzero(active&(~resident))
    if policy==7:
        assert 1<=world<=8 and not substitution
        assignment=load_locality_near_assignment(demand,misses,owner,layer,200)
    elif policy==6:
        assert 1<=world<=8 and not substitution
        assignment=load_fanout_assignment(demand,effective,lengths,org,owner,primary,layer,experts,misses,world)
    elif policy==5:
        assert 1<=world<=8 and not substitution
        assignment=packet_fanout_assignment(effective,lengths,org,primary,layer,experts,misses,world)
    elif policy==4:
        assert 1<=world<=8 and not substitution
        assignment=load_assignment(demand,misses,owner,layer)
    elif policy==3:
        assert not substitution
        assignment=fanout_assignment(effective,lengths,org,primary,layer,experts,misses,world)
    else:
        assignment=balanced_assignment(demand,misses,world,policy==0)
    for i in range(len(misses)):rank_fetches[assignment[i]]+=1
    row[44]=rank_fetches.max();row[45]=rank_fetches.min();assert row[44]-row[45]<=1
    for i in range(len(misses)):
        e=misses[i];r=assignment[i];key=layer*experts+e
        slot=choose_slot(r,layer,active,slots,capacities,last,gates,gate_eviction,experts);assert slot>=0
        fetches.append((r,key,slot,int(slots[r,slot]),0));place(r,key,slot,False,tick,slots,owner,primary,last,seen,lost,birth,reuses,row)
    if policy==2:
        for i in range(len(misses)):
            e=misses[i];key=layer*experts+e;best=-1;count=-1
            for r in range(world):
                if r!=primary[key] and future[e,r]>count:best=r;count=future[e,r]
            value=count*2*ROW_BYTES
            bucket=0 if value<EXPERT_BYTES/4 else 1 if value<EXPERT_BYTES/2 else 2 if value<EXPERT_BYTES else 3 if value<2*EXPERT_BYTES else 4
            row[38+bucket]+=1
            if value>=EXPERT_BYTES:
                slot=choose_slot(best,layer,active,slots,capacities,last,gates,gate_eviction,experts)
                if slot<0:row[43]+=1
                else:
                    fetches.append((best,key,slot,int(slots[best,slot]),1));place(best,key,slot,True,tick,slots,owner,primary,last,seen,lost,birth,reuses,row)
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

    destinations=np.full((n,topk),-1,np.int8)
    for t in range(n):
        for j in range(lengths[t]):
            key=layer*experts+effective[t,j];r=origins[t]
            destinations[t,j]=r if owner[key]&(1<<r) else primary[key]
    return targets,effective,masses,lengths,destinations,fetches,row

class Policy:
    def __init__(self,capacities,similarity,substitution,policy,seed=42):
        self.capacities=np.asarray(capacities,np.int32);self.similarity=similarity
        self.substitution=substitution;self.policy=policy;w=len(capacities);l,e=similarity.shape[:2];k=l*e
        self.slots=np.full((w,max(capacities)),-1,np.int32);self.owner=np.zeros(k,np.int16);self.primary=np.full(k,-1,np.int8)
        self.last=np.zeros((w,k),np.int32);self.seen=np.zeros(k,np.bool_);self.lost=np.zeros(k,np.bool_)
        self.birth=np.full((w,k),-1,np.int32);self.reuses=np.zeros((w,k),np.int32);self.gates=np.zeros((l,e),np.float32)
        seed_rng(seed)
    def apply(self,event,selected,weights,origins,gate_scores,future):
        return step(event,selected,weights,origins,gate_scores,self.similarity,self.capacities,self.substitution,self.policy,future,self.slots,self.owner,self.primary,self.last,self.seen,self.lost,self.birth,self.reuses,self.gates)
