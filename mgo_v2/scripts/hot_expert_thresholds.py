#!/usr/bin/env python3
"""H1 microcost reduction and H2/H4 mapping, without extrapolation."""
from hot_expert_cpu import P,ROOT,sha,write,csvfile,receipt
import gzip,json,resource
from pathlib import Path
import numpy as np
N=(1,2,4,8,16,32,64,128,256,512)

def crossing(costs,h2d):
    return next((n for n in N if costs[n]>=h2d),None)
def stats(values):return dict(median_ms=float(np.median(values)),p90_ms=float(np.percentile(values,90)))

def main():
    state=json.loads((P/'H1.json').read_text());assert state['status']=='PASS'
    for r in state['receipts']:assert Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256']
    allcells={};micro=[];pooled={};h2d={}
    for cell in state['cells']:
        label=cell['label'];ranks=[json.loads((ROOT/'H1'/label/f'rank{r}.json').read_text()) for r in range(4)]
        assert all(r['status']=='PASS' and r['rank']==i for i,r in enumerate(ranks))
        if cell['kind']=='smoke':
            status=json.loads((ROOT/'H1'/label/'status.json').read_text());assert status['status']=='PASS'
            assert status['transports']==['P2P/IPC'] if cell['mode']=='T0' else all(s.startswith('SHM/') for s in status['transports'])
            continue
        for r in ranks:
            assert r['no_model'] and r['warmups']==10 and r['samples']==30 and r['payload_valid']
            expected={'NCCL_CUMEM_ENABLE':'0'}
            if cell['mode']=='R3':expected.update(NCCL_P2P_LEVEL='LOC',NCCL_IB_DISABLE='1')
            assert r['transport_env']==expected
        cases=[r['case'] for r in ranks[0]['records']]
        assert cases==(list(N) if cell['kind']=='pair' else ['single-rank','four-rank-concurrent'])
        for i,case in enumerate(cases):
            arrays=[r['records'][i]['cuda_ms'] for r in ranks]
            assert all(len(a)==30 for a in arrays)
            values=np.max(np.array(arrays),axis=0).tolist();assert all(v>0 for v in values)
            row=dict(kind=cell['kind'],mode=cell['mode'],pass_id=cell.get('pass_id'),case=case,samples=30,**stats(values))
            sample_key=f"pair_pass{cell['pass_id']}_{cell['mode']}" if cell['kind']=='pair' else label
            micro.append(row);allcells[sample_key,str(case)]=values
            if cell['kind']=='pair':pooled.setdefault((cell['mode'],case),[]).extend(values)
            else:h2d[case]=row
    for (mode,n),values in pooled.items():
        assert len(values)==60
        micro.append(dict(kind='pair-pooled',mode=mode,pass_id=None,case=n,samples=60,**stats(values)))
    csvfile(P/'break_even_microcost.csv',micro)
    thresholds=[]
    for mode in ('T0','R3'):
        for condition in ('single-rank','four-rank-concurrent'):
            for metric in ('median','p90'):
                costs={n:stats(pooled[mode,n])[metric+'_ms'] for n in N};bound=h2d[condition][metric+'_ms']
                star=crossing(costs,bound)
                per_pass={str(p):crossing({n:stats(allcells[f'pair_pass{p}_{mode}',str(n)])[metric+'_ms'] for n in N},bound) for p in (0,1)}
                thresholds.append(dict(mode=mode,h2d_condition=condition,metric=metric,n_star=star,n_star_display=str(star) if star is not None else '>512',h2d_ms=bound,remote_at_crossing_ms=costs[star] if star else None,per_pass_n_star=per_pass,nonmonotonic_adjacent_pairs=sum(costs[b]<costs[a] for a,b in zip(N,N[1:]))))
    write(P/'break_even_microcost.json',dict(status='PASS',rows=micro,thresholds=thresholds,raw_receipts=state['receipts'],aggregation='Per-iteration max across four ranks; pooled 60 samples per n/mode. H2D 30 samples per condition.',excluded='Cache-victim and full-model scheduling costs; thresholds are optimistic isolated lower bounds.'))
    coverage=[];eligible=[];h0=json.loads((P/'H0.json').read_text());maxima={};p99={}
    for batch in (8,16,32):
        rec=next(r for r in h0['receipts'] if f'B{batch}.' in r['path']);assert sha(rec['path'])==rec['sha256']
        with gzip.open(rec['path'],'rt') as f:rows=[r for line in f if (r:=json.loads(line))['remote']]
        totalroutes=sum(r['n'] for r in rows);totalpeer=sum(r['peer_bytes_saved'] for r in rows)
        maxima[batch]=max(r['n'] for r in rows);p99[batch]=float(np.percentile([r['n'] for r in rows],99))
        for t in thresholds:
            n=t['n_star'];chosen=[r for r in rows if n is not None and r['n']>=n]
            layers={str(l):sum(r['layer']==l for r in chosen) for l in range(48)}
            coverage.append(dict(batch=batch,mode=t['mode'],h2d_condition=t['h2d_condition'],metric=t['metric'],n_star=t['n_star_display'],remote_occurrences=len(rows),crossing_occurrences=len(chosen),candidate_fraction=len(chosen)/len(rows),remote_route_fraction=sum(r['n'] for r in chosen)/totalroutes,marginal_peer_fraction=sum(r['peer_bytes_saved'] for r in chosen)/totalpeer,distinct_pairs=len({(r['layer'],r['expert'],r['rank']) for r in chosen}),layer_distribution=layers))
            if chosen and t['h2d_condition']=='four-rank-concurrent':eligible.append((batch,n))
    csvfile(P/'threshold_coverage.csv',coverage);write(P/'threshold_coverage.json',dict(status='PASS',rows=coverage,marginal_denominator='Sum of individual exact savings; not jointly attainable wire bytes.'))
    t0=next(t['n_star'] for t in thresholds if (t['mode'],t['h2d_condition'],t['metric'])==('T0','single-rank','median'))
    r3=next(t['n_star'] for t in thresholds if (t['mode'],t['h2d_condition'],t['metric'])==('R3','single-rank','median'))
    monotonic=all(maxima[a]<=maxima[b] and p99[a]<=p99[b] for a,b in ((8,16),(16,32)))
    covered=next(r['remote_route_fraction'] for r in coverage if (r['batch'],r['mode'],r['h2d_condition'],r['metric'])==(32,'R3','single-rank','median'))
    if t0 is None and r3 is None:decision='NO_TRANSPORT_CROSSOVER'
    elif r3 is None:decision='NO_MEASURED_R3_CROSSOVER'
    elif covered>0 or (2*maxima[32]>=r3 and monotonic):decision='LARGE_BATCH_CAPTURE_WORTHWHILE'
    elif 2*maxima[32]<r3:decision='CURRENT_BATCH_TOO_COLD'
    else:decision='LARGE_BATCH_GATE_NOT_MET'
    cells=[dict(batch=b,threshold=n) for b,n in sorted(set(eligible))];assert len(cells)<=12
    write(P/'H2_H4.json',dict(status='PASS',decision=decision,thresholds=thresholds,H3_cells=cells,H3_h2d_condition='four-rank-concurrent',B32_max_remote=maxima[32],max_remote_by_batch=maxima,p99_remote_by_batch=p99,monotonic_hotness=monotonic,R3_optimistic_B32_route_coverage=covered,new_B64_capture=False,new_B128_capture=False,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    print(json.dumps(dict(decision=decision,H3_cells=cells,thresholds=thresholds),indent=2))
if __name__=='__main__':main()
