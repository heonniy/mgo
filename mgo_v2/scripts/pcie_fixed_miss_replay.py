"""Conditional Stage-II comparison: restore identical pre-event cache/input."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
PKG=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PKG));import mgo_v2
from env_offload_policy import Policy
from pcie_host import write


def replay(capture,reference,cost_path,out):
    out.mkdir(parents=True,exist_ok=False)
    with np.load(capture) as archive:source={key:archive[key] for key in archive.files}
    expected=np.load(reference)
    costs=np.asarray(json.loads(cost_path.read_text())['matrix'],np.int64)
    assert len(source['events'])==len(expected)
    arms={'G-BR':(0,None),'G-CA':(1,None),'G-NUMA-CA':(1,costs),'G-NEAR':(7,None)}
    policies={name:Policy(source['capacities'],np.zeros((48,128,128),np.float32),False,kind,42,
                         native_pcie=True,quota_mode='group_balanced',peer_costs=cost) for name,(kind,cost) in arms.items()}
    timings={name:[] for name in arms};assignments=[]
    for i,event in enumerate(source['events']):
        lo,hi=source['offsets'][i:i+2]
        inputs=[source[k][lo:hi] for k in ('selected','weights','origins')]+[source['gate'][i]]
        baseline_quota=None;baseline_keys=None
        for name,p in policies.items():
            for field in ('slots','last','gates','seen'):np.copyto(getattr(p,field),source[field][i])
            p.owner.fill(0);p.primary.fill(-1);p.lost.fill(False);p.birth.fill(-1);p.reuses.fill(0)
            keys=[]
            for rank,slots in enumerate(p.slots):
                valid=slots[slots>=0];keys.extend(valid.tolist());p.owner[valid]=1<<rank;p.primary[valid]=rank
            assert len(keys)==len(set(keys))
            start=time.perf_counter_ns();result=p.apply(int(event),*inputs,None)
            timings[name].append((time.perf_counter_ns()-start)/1000)
            fetched=np.asarray(result[5]);quota=p.native_pcie.quotas.copy()
            assignments.extend([name,int(event),int(event)%48,*map(int,fetch)] for fetch in fetched)
            miss_keys=fetched[:,1] if len(fetched) else np.empty(0,np.int64)
            if baseline_quota is None:baseline_quota=quota;baseline_keys=miss_keys
            np.testing.assert_array_equal(quota,baseline_quota)
            np.testing.assert_array_equal(miss_keys,baseline_keys)
            assert len(fetched)==expected[i,19] and np.all(p.owner>=0)
            assert np.all((p.owner==0)|((p.owner&(p.owner-1))==0))
            if name=='G-NEAR':np.testing.assert_array_equal(result[-1],expected[i,:48])
    results={}
    with (out/'assigned_experts.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['arm','event','layer','rank','expert_key','logical_slot','victim','replica'])
        writer.writerows(assignments)
    for name,p in policies.items():
        trace=p.native_pcie.trace();np.save(out/f'{name}_trace.npy',trace)
        if name=='G-NEAR':np.testing.assert_array_equal(trace,expected)
        decode=trace[source['events']>=48]
        results[name]=dict(events=len(trace),decode_events=len(decode),
                          h2d_fetches=int(decode[:,19].sum()),h2d_bytes=int(decode[:,19].sum())*9437184,
                          cross_group_expert_routes=int(decode[:,53].sum()),within_group_expert_routes=int(decode[:,54].sum()),
                          cross_group_token_rank_pairs=int(decode[:,55].sum()),within_group_token_rank_pairs=int(decode[:,56].sum()),
                          forward_cross_group_activation_bytes=int(decode[:,55].sum())*4096,
                          projected_critical_rank_rows_mean=float(decode[:,57:61].max(axis=1).mean()),
                          adapter_plus_controller_us_mean=float(np.mean(timings[name])),
                          adapter_plus_controller_us_median=float(np.median(timings[name])))
        p.native_pcie.close()
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        return h.hexdigest()
    receipt=dict(status='PASS',scope='Conditional current-event placement on the captured G-NEAR cache trajectory; future cache states restored, not independent model generation',
                 capture=str(capture),capture_sha256=sha(capture),reference=str(reference),reference_sha256=sha(reference),
                 measured_costs=costs.tolist(),identical_misses_and_per_rank_quotas_every_event=True,reference_near_all_outputs_exact=True,
                 exact_only=True,main_capacities=source['capacities'].tolist(),physical_cache_slots=1843,arms=results)
    write(out/'result.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--costs',type=Path,default=PKG/'experiments/pcie_topology_ablation_20261009/microbench_physical_cores/peer_costs.json');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();replay(a.capture,a.reference,a.costs,a.out)
