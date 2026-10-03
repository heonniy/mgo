#!/usr/bin/env python3
"""CPU-only hotness accounting on hash-verified existing exact captures."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import resource
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
import csv,gzip,hashlib,json,math,sys
from collections import Counter
from pathlib import Path
import numpy as np
from replica_pareto_cpu import ROW_BYTES
from run_rank_local_reuse import receipt,sha,write,csvfile
P=Path(__file__).resolve().parents[1]/'experiments/hot_expert_replication_threshold_20261003'
PREV=P.parent/'rank_local_reuse_oracle_20261003'
ROOT=Path('/home/hwlee/mgo-results/hot_expert_replication_threshold_20261003')

def current_pairs(event):
    origins=np.array(event['origins']);selected=np.array(event['selected']);dest=np.array(event['destinations'])
    before={tuple(k):r for k,r in event['cache_before'][2]}
    for e in np.unique(selected):
        for r in range(4):
            ts,ks=np.where((selected==e)&(origins[:,None]==r))
            n=len(ts)
            if not n:continue
            owners=dest[ts,ks];assert len(set(owners))==1
            owner=int(owners[0]);remote=owner!=r
            shared=int(((dest[ts]==owner).sum(axis=1)>1).sum())
            dispatch=(n-shared)*ROW_BYTES if remote else 0
            combine=n*ROW_BYTES if remote else 0
            assert dispatch+combine<=2*n*ROW_BYTES
            key=(event['layer'],int(e))
            yield dict(step=event['step'],layer=event['layer'],expert=int(e),rank=r,n=n,
                serving_owner=owner,remote=remote,local=not remote,
                pre_event_global_resident=key in before,pre_event_local_resident=before.get(key)==r,
                shared_owner_routes=shared,dispatch_bytes_saved=dispatch,
                combine_bytes_saved=combine,peer_bytes_saved=dispatch+combine)

def quantiles(values):
    return {f'p{q}':float(np.percentile(values,q)) for q in (50,90,95,99)}|{'max':int(max(values))}

def h0():
    ROOT.mkdir(exist_ok=True);assert not (P/'H0.json').exists(),'No extra runs'
    distributions=[];savings=[];thresholds=[];concentrations=[];receipts=[];provenance=[]
    for batch in (8,16,32):
        prev=json.loads((PREV/f'R02_B{batch}.json').read_text());assert prev['status']=='PASS'
        for r in prev['provenance']['raw_receipts']:
            assert Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256']
        summary=prev['provenance']['summary'];assert sha(summary['path'])==summary['sha256']
        assert prev['provenance']['cross_rank_validation']
        rec=next(r for r in prev['artifacts'] if f'F_B{batch}_events' in r['path']);assert sha(rec['path'])==rec['sha256']
        rows=[];outpath=ROOT/f'hotness_B{batch}.jsonl.gz'
        with gzip.open(rec['path'],'rt') as src,gzip.open(outpath,'wt') as out:
            for i,line in enumerate(src):
                event=json.loads(line);assert event['event']==i+48
                for row in current_pairs(event):
                    assert row['n']<=batch
                    rows.append(row);out.write(json.dumps(row,separators=(',',':'))+'\n')
        assert i==383
        remote=[r for r in rows if r['remote']]
        for label,subset in [('all',rows),('remote',remote)]:
            distributions.append(dict(batch=batch,subset=label,occurrences=len(subset),**quantiles([r['n'] for r in subset])))
        total_routes=sum(r['n'] for r in remote);total_marginal=sum(r['peer_bytes_saved'] for r in remote)
        assert total_routes==prev['baseline']['remote_expert_routes']
        for n in (1,2,4,8,16,32,64,128):
            sub=[r for r in remote if r['n']>=n]
            thresholds.append(dict(batch=batch,n_threshold=n,occurrences=len(sub),candidate_fraction=len(sub)/len(remote),route_fraction=sum(r['n'] for r in sub)/total_routes,marginal_peer_fraction=sum(r['peer_bytes_saved'] for r in sub)/total_marginal))
        for n in sorted({r['n'] for r in remote}):
            sub=[r for r in remote if r['n']==n]
            savings.append(dict(batch=batch,n=n,occurrences=len(sub),remote_routes=n*len(sub),dispatch_bytes_saved=sum(r['dispatch_bytes_saved'] for r in sub),combine_bytes_saved=sum(r['combine_bytes_saved'] for r in sub),peer_bytes_saved=sum(r['peer_bytes_saved'] for r in sub),shared_owner_routes=sum(r['shared_owner_routes'] for r in sub)))
        groups={}
        for r in remote:
            key=r['layer'],r['expert'],r['rank'];v=groups.setdefault(key,[0,0]);v[0]+=r['n'];v[1]+=r['peer_bytes_saved']
        ranked=sorted(groups.items(),key=lambda x:(-x[1][0],x[0]))
        for f in (.01,.05,.1,.2):
            top=ranked[:math.ceil(len(ranked)*f)]
            concentrations.append(dict(batch=batch,top_pair_fraction=f,pairs=len(top),remote_route_share=sum(v[0] for _,v in top)/total_routes,marginal_peer_share=sum(v[1] for _,v in top)/total_marginal))
        counts={str(n):sum(r['peer_bytes_saved']>=n for r in remote) for n in (65536,131072,262144,524288,1048576,2097152,4194304,8388608)}
        provenance.append(dict(batch=batch,source=prev['provenance'],F_archive=rec,actual_F_peer_bytes=prev['baseline']['peer_activation_bytes'],sum_individual_marginal_bytes=total_marginal,saving_threshold_candidate_counts=counts))
        receipts.append(receipt(outpath))
        print(json.dumps(dict(stage='H0',batch=batch,remote_hotness=quantiles([r['n'] for r in remote]))),flush=True)
    csvfile(P/'hotness_distribution.csv',distributions);write(P/'hotness_distribution.json',dict(rows=distributions,threshold_coverage=thresholds,top_pairs=concentrations))
    csvfile(P/'hotness_peer_saving.csv',savings);write(P/'hotness_peer_saving.json',dict(rows=savings,provenance=provenance,occurrence_receipts=receipts,note='Per-occurrence marginal dispatch savings interact; their sum is not jointly attainable wire traffic. Hot pairs rank by total remote route demand across eight decode steps.'))
    assert 'torch' not in sys.modules
    write(P/'H0.json',dict(status='PASS',batches=[8,16,32],receipts=receipts,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,cpu_only=True,address_space_limit_bytes=4*1024**3))
if __name__=='__main__':h0()
