"""Small independent full-traffic oracle checks; no production trace runs."""
import copy,json
from pathlib import Path
import numpy as np
from cache_policy_cpu import CachePolicyReplay
from dynamic_refresh_cpu import DynamicReplay,FutureDemand,project,peer_rows,removal_rows
from replica_pareto_cpu import traffic


def packet(i,layer,raw,origins,weights=None):
    return dict(event=i,step=i//48,layer=layer,raw_selected_experts=np.array(raw),origin_ranks=np.array(origins),routing_weights=np.array(weights if weights is not None else np.ones_like(raw,dtype=float)*.1,dtype=np.float32),gate_scores=np.zeros(128,dtype=np.float32))


def fixture(policy,sub,seed):
    rng=np.random.default_rng(seed);sim=rng.uniform(.3,.9,(48,128,128)).astype(np.float32)
    events=[]
    for step in range(1,7):
        for layer in (0,1):
            raw=np.array([rng.choice(8,3,False) for _ in range(6)])
            if layer==0:raw[0]=[0,1,3]
            events.append(packet(step*48+layer,layer,raw,[0,0,0,1,1,1]))
    future=FutureDemand(events) if policy in ('O4','OR') else None
    b=DynamicReplay([5,5],.2,'lru',sim,sub,policy,future)
    b.tick=1;b.begin_event(set())
    for rank,key,slot in [(1,(0,0),0),(0,(0,1),0),(1,(0,1),1),(0,(1,2),1),(1,(1,2),2),(1,(0,3),3)]:b.place(rank,key,(slot,None))
    b.begin_event({(0,0),(0,3)})
    return b,events


def bytes_of(p):return traffic(p['origins'],np.split(p['dest'],np.cumsum(p['lengths'])[:-1]),p['occupancy'].shape[1])['peer_bytes']

def brute_best(b,events,current):
    options=[];layer=0
    for e in sorted(set(current['selected'])):
        for r in range(2):
            if r in b.owners[layer,int(e)] or not np.any((current['selected']==e)&(current['origins'][current['tokens']]==r)):continue
            for v in sorted(b.ranks[r]):
                if v in b.active or len(b.owners[v])<2:continue
                after=copy.deepcopy(b);slot=after.ranks[r][v].slot
                DynamicReplay.place(after,r,(layer,int(e)),(slot,v))
                cur_after=project(after,0,current['origins'],np.split(current['selected'],np.cumsum(current['lengths'])[:-1]))
                gain=bytes_of(current)-bytes_of(cur_after)
                if b.refresh_policy in ('O4','OR'):
                    for l in {0,v[0]}:
                        future=[x for x in events if x['layer']==l and x['event']>b.tick-1]
                        if b.refresh_policy=='O4':future=future[:4]
                        for item in future:
                            origin,eff=b.future.effective(b,item)
                            origin2,eff2=after.future.effective(after,item)
                            assert [list(x) for x in eff]==[list(x) for x in eff2]
                            gain+=bytes_of(project(b,l,origin,eff))-bytes_of(project(after,l,origin2,eff2))
                options.append((-gain,int(e),r,b.ranks[r][v].used,v))
    good=[v for v in options if v[0]<0]
    return min(good) if good else None


def main():
    checks=[];rng=np.random.default_rng(221)
    for rho in (0,.125,.25,.5,.75):
        sim=np.ones((48,128,128),dtype=np.float32)
        a=CachePolicyReplay([18]*4,rho,'lru',sim);b=DynamicReplay([18]*4,rho,'lru',sim,False,'N0')
        for i in range(60):
            origins=np.repeat(np.arange(4),3);selected=np.array([rng.choice(12,3,False) for _ in origins])
            x=a.event(i%5,origins,selected);y=b.event(i%5,origins,selected)
            assert a.state()==b.state() and x[3]==y[3]
            assert all(y[0][k]==v for k,v in x[0].items())
        checks.append(f'N0 rho={rho}: 60 exact action/counter/state matches')
    swaps=0
    for policy in ('C1','C2','O4','OR'):
        for sub in (False,True):
            for seed in range(4):
                b,events=fixture(policy,sub,seed)
                current=project(b,0,np.array([0,0,1,1]),[[0,3],[0],[0],[3]])
                # Instrument only accepted swaps; recompute a slow independent
                # full-traffic best swap at the current physical cache state.
                original=b.place
                def checked(rank,key,choice):
                    p=project(b,0,np.array([0,0,1,1]),[[0,3],[0],[0],[3]])
                    expected=brute_best(b,events,p);assert expected is not None
                    assert (key[1],rank,choice[1])==(expected[1],expected[2],expected[4]),(policy,sub,seed,key,rank,choice,expected)
                    original(rank,key,choice)
                b.place=checked
                loc={(int(e),r):np.flatnonzero((current['selected']==e)&(current['origins'][current['tokens']]==r)) for e in np.unique(current['selected']) for r in range(2)}
                ops,_=b.refresh(0,current['origins'],current['selected'],current['tokens'],current['lengths'],current['dest'],current['occupancy'],loc)
                swaps+=len(ops);assert len(ops)<=(2 if policy=='C2' else 1)
                if not ops:assert brute_best(b,events,current) is None
                assert b.duplicates==2 and all(len(b.owners[k])>0 for k in b.owners)
        checks.append(policy+': exact selected swap against enumerated before/after traffic, OFF/ON, primary promotion and same-layer coalescing')
    assert swaps>0
    b=DynamicReplay([4]*4,.25,'lru',np.ones((48,128,128),dtype=np.float32),False,'N0')
    b.tick=1;b.begin_event(set())
    for r,k,slot in [(1,(0,0),0),(0,(0,1),0),(2,(0,1),0),(3,(0,1),0),(0,(0,3),1)]:b.place(r,k,(slot,None))
    origins=np.array([0,1,2,3,0,1,2,3]);effective=[[1,3],[1,3],[1,0],[1,0]]*2
    before=project(b,0,origins,effective)
    for rank in (0,2,3):
        after=copy.deepcopy(b);after.owners[0,1].remove(rank)
        if after.primary[0,1]==rank:after.primary[0,1]=min(after.owners[0,1])
        expected=bytes_of(before)-bytes_of(project(after,0,origins,effective))
        assert removal_rows(before,b,0,1,rank)*4096==expected
    checks.append('R4 three-copy victim: primary promotion and remote-origin dispatch changes match full traffic')
    p=Path(__file__).resolve().parents[1]/'experiments/dynamic_replica_refresh_20261003/policy_tests.json'
    p.write_text(json.dumps(dict(status='PASS',checks=checks,independently_checked_swaps=swaps),indent=2)+'\n');print(p.read_text())
if __name__=='__main__':main()
