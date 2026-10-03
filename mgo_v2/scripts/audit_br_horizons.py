#!/usr/bin/env python3
"""Read-only raw routing audit of two prefixes from one master capture."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003')
P=Path(__file__).resolve().parents[1]/'experiments/br_ca_carep_cpu_headroom_20261003'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()
def quantiles(x):return {name:float(np.quantile(x,q)) for name,q in [('p50',.5),('p90',.9),('p99',.99),('max',1)]}
def main(name):
    selected=[];weights=[];receipts=[]
    for rank in range(8):
        path=ROOT/'captures'/name/f'rank{rank}';r=json.loads((path/'receipt.json').read_text());assert r['status']=='PASS' and r['decode_forwards']==256 and r['request_ids']==list(range(rank,512,8))
        for f in r['files']:assert sha(path/f['file'])==f['sha256']
        selected.append(np.load(path/'decode_selected.npy',mmap_mode='r'));weights.append(np.load(path/'decode_weights.npy',mmap_mode='r'))
        assert selected[-1].shape==(256,48,64,8)
        receipts.append(dict(rank=rank,path=str(path/'receipt.json'),sha256=sha(path/'receipt.json')))
    rows=[];cross=[]
    for layer in range(48):
        counts=np.zeros((256,8,128),dtype=np.int32)
        for rank in range(8):
            for step in range(256):counts[step,rank]=np.bincount(selected[rank][step,layer].ravel(),minlength=128)
        assert np.all(counts.sum((1,2))==512*8)
        totals64=counts[:64].sum(0);totals_late=counts[64:].sum(0);seen=totals64>0;later=totals_late>0
        hot1=np.argsort(-totals64.ravel(),kind='stable')[:103];hot2=np.argsort(-totals_late.ravel(),kind='stable')[:103]
        cross.append(dict(dataset=name,layer=layer,seen_first64=int(seen.sum()),seen_first64_recur_after64=int((seen&later).sum()),fraction_first64_pairs_recur_after64=float((seen&later).sum()/seen.sum()),hot_set_size=103,hot_set_intersection=len(set(hot1)&set(hot2)),hot_set_jaccard=len(set(hot1)&set(hot2))/len(set(hot1)|set(hot2))))
        for horizon in (64,256):
            c=counts[:horizon];total=c.sum(0);flat=total.ravel();active=(c.sum(1)>0).sum(1)
            # Recurrence opportunities exclude the final horizon step.
            present=c>0;future=np.cumsum(present[::-1],axis=0)[::-1]-present;exposed=present[:-1]
            gaps=[]
            for rank,expert in zip(*np.where(total>0)):
                positions=np.flatnonzero(present[:,rank,expert]);gaps.extend(np.diff(positions).tolist())
            gate_mass=sum(float(w[:horizon,layer].sum(dtype=np.float64)) for w in weights)
            rows.append(dict(dataset=name,horizon=horizon,layer=layer,routes=int(c.sum()),gate_mass=gate_mass,active_unique_experts_mean=float(active.mean()),**{'active_unique_experts_'+k:v for k,v in quantiles(active).items()},**{'rank_local_event_demand_'+k:v for k,v in quantiles(c).items()},**{'rank_local_cumulative_demand_'+k:v for k,v in quantiles(flat).items()},top10_percent_pair_demand_share=float(np.sort(flat)[-103:].sum()/flat.sum()),max_to_mean_pair_demand=float(flat.max()/flat.mean()),observed_expert_rank_occurrences=int(present.sum()),occurrences_with_future_opportunity=int(exposed.sum()),occurrences_with_later_recurrence=int((exposed&(future[:-1]>0)).sum()),future_recurrence_fraction=float((exposed&(future[:-1]>0)).sum()/exposed.sum()),recurrence_gap_mean=float(np.mean(gaps)),recurrence_gap_p90=float(np.quantile(gaps,.9))))
    def table(filename,data):
        (P/(filename+'.json')).write_text(json.dumps(data,indent=2)+'\n')
        with (P/(filename+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
    table(name+'_horizon_layers',rows);table(name+'_horizon_cross64',cross)
    summary=dict(status='PASS',dataset=name,requests=512,logical_ranks=8,local_batch=64,new_generations_for_prefix=0,prefix_verified_by_single_array_slicing=True,rank_receipts=receipts,definitions=dict(demand='raw selected expert route count; all 1024 expert/rank pairs including zeros',top10='top ceil(.1*128*8)=103 pairs per layer, stable expert/rank index tie',recurrence='later same-layer raw rank demand; exclude final step from opportunity denominator',hot_overlap='top 103 cumulative expert/rank pairs at steps 1..64 versus 65..256'),horizons=[])
    for h in (64,256):
        rr=[r for r in rows if r['horizon']==h];routes=sum(r['routes'] for r in rr);assert routes==h*512*48*8
        summary['horizons'].append(dict(decode_steps=h,routes=routes,gate_mass=sum(r['gate_mass'] for r in rr),active_experts_mean=float(np.mean([r['active_unique_experts_mean'] for r in rr])),top10_share_mean=float(np.mean([r['top10_percent_pair_demand_share'] for r in rr])),recurrence_fraction=sum(r['occurrences_with_later_recurrence'] for r in rr)/sum(r['occurrences_with_future_opportunity'] for r in rr)))
    summary['seen64_pairs_recurring_later_fraction']=sum(r['seen_first64_recur_after64'] for r in cross)/sum(r['seen_first64'] for r in cross);summary['mean_hot_set_jaccard']=float(np.mean([r['hot_set_jaccard'] for r in cross]))
    (P/(name+'_horizon_audit.json')).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['MATH','ShareGPT'],required=True);main(parser.parse_args().dataset)
