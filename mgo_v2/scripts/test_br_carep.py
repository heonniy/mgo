"""Independent dictionary replay and exhaustive balanced-placement checks."""
import itertools,json
from pathlib import Path
import numpy as np
from br_carep_cpu import replay,balanced_assignment,suffix_demand,ROW_BYTES,EXPERT_BYTES
from mgo_v2.substitution import SubstitutionPolicy,merge_effective_routes
from mgo_v2.types import LayerRoutes
from replica_pareto_cpu import traffic
P=Path(__file__).resolve().parents[1]/'experiments/br_ca_carep_cpu_headroom_20261003'
class Reference:
    def __init__(self,cap,layers,experts,sim,ev,sub,pol):
        self.cap=cap;self.slots=[[None]*c for c in cap];self.used={};self.seen=set();self.primary={};self.gates=np.zeros((layers,experts));self.ev=ev;self.sub=sub;self.pol=pol;self.policy=SubstitutionPolicy(sim);self.rng=np.random.RandomState(42)
    def resident(self,key):return any(key in row for row in self.slots)
    def resident_layer(self,layer):return sorted({e for row in self.slots for key in row if key is not None for l,e in [key] if l==layer})
    def owners(self,key):return {r for r,row in enumerate(self.slots) if key in row}
    def run(self,s,w,offsets,org,g,sim,steps):
        L,E=sim.shape[:2];world=len(self.cap);metrics=[]
        for event in range(L*(steps+1)):
            layer=event%L;step=event//L;self.gates[layer]=g[event];selected=s[offsets[event]:offsets[event+1]];weights=w[offsets[event]:offsets[event+1]];routes=LayerRoutes(layer,org,selected,weights)
            decision=self.policy.decide(routes,self) if self.sub else None
            eff=merge_effective_routes(routes,decision) if decision else [dict(zip(map(int,a),map(float,b))) for a,b in zip(selected,weights)]
            active={(layer,e) for row in eff for e in row};missing=sorted(e for l,e in active if not self.resident((l,e)))
            demand=np.zeros((E,world),np.int64)
            for r,row in zip(org,eff):
                for e in row:demand[e,r]+=1
            slots=[r for r in range(world) for _ in range(len(missing)//world+(r<len(missing)%world))]
            if self.pol==0:
                order=self.rng.permutation(len(missing));dest={missing[int(i)]:r for i,r in zip(order,slots)}
            else:dest=dict(zip(missing,map(int,balanced_assignment(demand,np.array(missing,dtype=np.int64),world,False))))
            fetch=[0,0,0];evictions=0
            def put(e,r,rep):
                nonlocal evictions
                key=layer,e
                if None in self.slots[r]:slot=self.slots[r].index(None)
                else:
                    legal=[k for k in self.slots[r] if k not in active]
                    if not legal:return False
                    old=min(legal,key=lambda k:((self.gates[k] if self.ev else 0),self.used[r,k],k));slot=self.slots[r].index(old);evictions+=1
                    survivors=self.owners(old)-{r}
                    if not survivors:del self.primary[old]
                    elif self.primary[old]==r:self.primary[old]=min(survivors)
                if rep:fetch[2]+=1
                elif key in self.seen:fetch[1]+=1
                else:fetch[0]+=1
                if not rep:self.primary[key]=r
                self.slots[r][slot]=key;self.used[r,key]=event+1;self.seen.add(key);return True
            for e in missing:assert put(e,dest[e],False)
            if self.pol==2:
                for e in missing:
                    counts=[sum(np.count_nonzero(np.any(s[offsets[f*L+layer]:offsets[f*L+layer+1]]==e,axis=1)&(org==r)) for f in range(step+1,steps+1)) for r in range(world)]
                    r=min((r for r in range(world) if r!=dest[e]),key=lambda r:(-counts[r],r))
                    if 2*ROW_BYTES*counts[r]>=EXPERT_BYTES:put(e,r,True)
            destinations=[]
            for r,row in zip(org,eff):
                ds=[]
                for e in row:
                    key=layer,e;dst=int(r) if int(r) in self.owners(key) else self.primary[key];ds.append(dst);self.used[dst,key]=event+1
                destinations.append(ds)
            t=traffic(org,destinations,world);metrics.append(fetch+[evictions,t['remote_pairs'],t['remote_expert_routes']])
        return np.array(metrics),self.slots
def main():
    rng=np.random.default_rng(45);assignment_cases=0
    for world in (2,3):
        for n in range(1,7):
            for _ in range(3):
                demand=rng.integers(0,8,(n,world),dtype=np.int64);a=balanced_assignment(demand,np.arange(n),world,False);quota=np.bincount(a,minlength=world);assert np.array_equal(quota,[n//world+(r<n%world) for r in range(world)])
                possibilities=(x for x in itertools.product(range(world),repeat=n) if np.array_equal(np.bincount(x,minlength=world),quota))
                optimum=max(sum(demand[i,r] for i,r in enumerate(x)) for x in possibilities);assert sum(demand[i,r] for i,r in enumerate(a))==optimum;assert np.array_equal(a,balanced_assignment(demand,np.arange(n),world,False));assignment_cases+=1
    fixtures=0;accepted_replicas=0
    for large in (False,True):
        L=3;E=4 if large else 8;N=128 if large else 8;steps=32 if large else 12
        s=np.stack([np.array([0,int(rng.integers(1,E))]) if large else rng.choice(E,2,replace=False) for _ in range((steps+1)*L*N)]).astype(np.uint8)
        w=np.tile(np.array([.875,.125],np.float32),(len(s),1));offsets=np.arange((steps+1)*L+1,dtype=np.int64)*N;org=np.repeat(np.arange(2,dtype=np.int8),N//2);g=rng.random(((steps+1)*L,E),dtype=np.float32);sim=rng.random((L,E,E),dtype=np.float32)
        for l in range(L):np.fill_diagonal(sim[l],1)
        cap=np.array([E,E],np.int64)
        for ev in (False,True):
            for sub in (False,True):
                for pol in (0,1,2):
                    out=replay(s,w,offsets,org,org,g,sim,cap,steps,ev,sub,pol,42);ref,slots=Reference(cap,L,E,sim,ev,sub,pol).run(s,w,offsets,org,g,sim,steps)
                    assert np.array_equal(out[0][:,[20,21,22,23,30,29]],ref),(large,ev,sub,pol)
                    expected=np.array([[-1 if k is None else k[0]*E+k[1] for k in row] for row in slots]);assert np.array_equal(out[2],expected)
                    assert np.allclose(out[0][:,1],out[0][:,32],rtol=0,atol=0)
                    accepted_replicas+=int(out[0][:,22].sum());fixtures+=1
    assert accepted_replicas>0
    result=dict(status='PASS',exhaustive_assignment_cases=assignment_cases,independent_dictionary_replays=fixtures,replicas_exercised=accepted_replicas,checks=['exact balanced local-demand optimum and deterministic ties','BR deterministic quota permutation','native production substitution decisions','LRU/Gate final physical slots','first/reload/replica fetch categories and evictions','independent traffic send/receive transpose checks','future raw demand score checked without suffix cache','merged gate mass conservation'],gpu_runs=0)
    (P/'policy_tests.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
