#!/usr/bin/env python3
"""MAIN-only cache eviction replay over one frozen global routing trace.

The replay deliberately removes PREFETCH slots/actions so the measured quantity
is MAIN distinct-expert reuse caused by eviction alone. Admission remains the
current LA_CA_NEAR rule and capacities remain the production MAIN capacities.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

SCRIPTS=Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:sys.path.insert(0,str(SCRIPTS))
from la_placement import load_locality_near_assignment

EXPERTS=128
LAYERS=48
POLICIES=('gate_w128','lfu_reset','lfu_cumulative','lru_reset','lru_cumulative')

def _sha_slots(slots):
    return hashlib.sha256(np.asarray(slots,dtype=np.int32).tobytes()).hexdigest()

def _free_or_victim(policy,rank,layer,active,slots,capacities,last,gate_table,freq_epoch,freq_cum):
    for slot in range(int(capacities[rank])):
        if slots[rank,slot]<0:return slot
    best_slot=-1;best_score=None
    for slot in range(int(capacities[rank])):
        key=int(slots[rank,slot])
        if key//EXPERTS==layer and active[key%EXPERTS]:continue
        if policy=='gate_w128':
            score=(float(gate_table[key//EXPERTS,key%EXPERTS]),int(last[key]),key)
        elif policy.startswith('lfu_'):
            freq=freq_epoch[key] if policy=='lfu_reset' else freq_cum[key]
            score=(int(freq),int(last[key]),key)
        elif policy.startswith('lru_'):
            # A reload is itself an access, so standard LRU has no meaningful
            # reset-vs-persistent distinction. Both labels are retained because
            # the owner explicitly requested both variants.
            score=(int(last[key]),key)
        else:raise ValueError(policy)
        if best_score is None or score<best_score:best_score=score;best_slot=slot
    if best_slot<0:raise RuntimeError(f'rank {rank}: all MAIN slots pinned by the current event')
    return best_slot

def replay(histograms,gates,capacities,policy):
    assert policy in POLICIES
    histograms=np.asarray(histograms,dtype=np.int64);gates=np.asarray(gates,dtype=np.float32)
    assert histograms.ndim==3 and histograms.shape[1:]==(4,EXPERTS)
    assert gates.shape==(len(histograms),EXPERTS)
    assert len(histograms)%LAYERS==0 and len(histograms)>=2*LAYERS
    capacities=np.asarray(capacities,dtype=np.int32);assert capacities.shape==(4,)
    keys=LAYERS*EXPERTS
    slots=np.full((4,int(capacities.max())),-1,np.int32)
    owner=np.zeros(keys,np.int16) # one-hot rank bit; replication is disabled
    last=np.zeros(keys,np.int64)
    freq_epoch=np.zeros(keys,np.int64)
    freq_cum=np.zeros(keys,np.int64)
    seen=np.zeros(keys,np.bool_)
    gate_table=np.zeros((LAYERS,EXPERTS),np.float32)
    first_fetches=reload_fetches=evictions=0
    step_rows={}
    total=dict(uses=0,hits=0,token_uses=0,token_hits=0,misses=0)
    for event in range(len(histograms)):
        layer=event%LAYERS;tick=event+1;step=event//LAYERS
        gate_table[layer]=gates[event]
        demand=histograms[event].T.copy() # [expert, origin-rank]
        totals=demand.sum(axis=1)
        active=totals>0
        active_ids=np.flatnonzero(active)
        base=layer*EXPERTS
        resident=np.array([owner[base+int(e)]!=0 for e in active_ids],dtype=np.bool_)
        hits=int(resident.sum());uses=int(len(active_ids))
        token_uses=int(totals[active_ids].sum())
        token_hits=int(totals[active_ids[resident]].sum()) if hits else 0
        misses=active_ids[~resident]
        if step>=1:
            row=step_rows.setdefault(step,dict(uses=0,hits=0,token_uses=0,token_hits=0,misses=0,evictions=0,reloads=0))
            row['uses']+=uses;row['hits']+=hits;row['token_uses']+=token_uses;row['token_hits']+=token_hits;row['misses']+=len(misses)
            total['uses']+=uses;total['hits']+=hits;total['token_uses']+=token_uses;total['token_hits']+=token_hits;total['misses']+=len(misses)
        if len(misses):
            assignment=load_locality_near_assignment(demand,misses,owner,layer,200)
            quotas=np.bincount(np.asarray(assignment,dtype=np.int64),minlength=4)
            assert int(quotas.max()-quotas.min())<=1
            for e,r in zip(misses,assignment):
                e=int(e);r=int(r);key=base+e
                slot=_free_or_victim(policy,r,layer,active,slots,capacities,last,gate_table,freq_epoch,freq_cum)
                victim=int(slots[r,slot])
                if victim>=0:
                    owner[victim]=0;freq_epoch[victim]=0;evictions+=1
                    if step>=1:step_rows[step]['evictions']+=1
                if seen[key]:
                    reload_fetches+=1
                    if step>=1:step_rows[step]['reloads']+=1
                else:first_fetches+=1
                seen[key]=True;slots[r,slot]=key;owner[key]=np.int16(1<<r);last[key]=tick;freq_epoch[key]=0
        # One LFU access per distinct expert event, matching the requested
        # distinct-expert hit objective rather than token-weighted popularity.
        for e in active_ids:
            key=base+int(e)
            assert owner[key]!=0
            last[key]=tick;freq_epoch[key]+=1;freq_cum[key]+=1
    for row in step_rows.values():
        row['hit_rate']=row['hits']/row['uses'] if row['uses'] else 0.0
        row['token_weighted_hit_rate']=row['token_hits']/row['token_uses'] if row['token_uses'] else 0.0
    total['hit_rate']=total['hits']/total['uses']
    total['token_weighted_hit_rate']=total['token_hits']/total['token_uses']
    total.update(first_fetches=int(first_fetches),reload_fetches=int(reload_fetches),evictions=int(evictions),
                 final_occupancy=int(np.count_nonzero(slots>=0)),final_slots_sha256=_sha_slots(slots))
    first=step_rows[1]
    return dict(policy=policy,all_decode=total,first_decode_step=first,
                per_step={str(k):v for k,v in sorted(step_rows.items())})

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trace',type=Path,required=True);p.add_argument('--meta',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    meta=json.loads(a.meta.read_text());assert meta['status']=='PASS'
    z=np.load(a.trace);hist=z['histograms'];gates=z['gates'];events=z['events']
    assert np.array_equal(events,np.arange(len(events),dtype=events.dtype))
    assert len(events)==meta['events']==48*meta['output_tokens']
    capacities=np.asarray(meta['main_capacities'],dtype=np.int32)
    results={name:replay(hist,gates,capacities,name) for name in POLICIES}
    # Under standard LRU semantics a miss/reload is an access at the current
    # tick, hence reset and persistent-history variants must be identical.
    assert results['lru_reset']==dict(results['lru_cumulative'],policy='lru_reset')
    ranked=sorted(POLICIES,key=lambda n:(-results[n]['all_decode']['hit_rate'],
                                        results[n]['all_decode']['reload_fetches'],n))
    out=dict(status='PASS',trace_meta=meta,policies=results,ranking_by_main_distinct_hit=ranked,
             best_by_main_distinct_hit=ranked[0],
             lru_reset_vs_cumulative='IDENTICAL_BY_STANDARD_LRU_SEMANTICS',
             lfu_frequency_unit='one access per distinct layer/step expert; not token count',
             prefetch_simulated=False,admission='LA_CA_NEAR',near_bps=200)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',best=ranked[0],
        rates={k:results[k]['all_decode']['hit_rate'] for k in POLICIES}),sort_keys=True))

if __name__=='__main__':main()
