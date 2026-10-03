#!/usr/bin/env python3
"""Exact resource comparisons and bounded refresh diagnostics, no timing model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import resource
resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
import bisect,csv,hashlib,json,math
from collections import Counter
from fractions import Fraction
from pathlib import Path
import numpy as np
from price_envelope import fraction_record,serial_envelope
P=Path(__file__).resolve().parents[1]/'experiments/dynamic_replica_refresh_20261003'
OLD=P.parent/'cache_eviction_substitution_20261003'
ROOT=Path('/home/hwlee/mgo-results/dynamic_replica_refresh_20261003')
OLDROOT=Path('/home/hwlee/mgo-results/cache_eviction_substitution_20261003')
H='expert_h2d_bytes';T='peer_activation_bytes'
POLICIES=('N0','C1','C2','O4','OR');PRICES=(32,64,96,128,256)
BINS=(0,1,8,16,32,48,96,192,100000)

def write(name,obj):(P/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def table(name,rows):
    write(name+'.json',rows)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (P/(name+'.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def meta(c):return {k:c[k] for k in ('id','batch','cache_ratio','eviction','substitution','rho','policy','duplicate_cap')}
def group(c):return tuple(c[k] for k in ('batch','cache_ratio','eviction','substitution','rho'))
def flat(c):
    out=meta(c)
    for scope in ('full','decode'):out.update({scope+'_'+k:v for k,v in c[scope].items()})
    return out
def dominates(a,b):return a[H]<=b[H] and a[T]<=b[T] and (a[H]<b[H] or a[T]<b[T])
def ratio(a,b):return a/b if b else None

def price_case(dh,dt):
    if dh==dt==0:return dict(relation='equal',lambda_refresh=None,favors=None)
    if dh<=0 and dt<=0:return dict(relation='refresh_dominates',lambda_refresh=None,favors='all nonnegative prices, with possible boundary tie')
    if dh>=0 and dt>=0:return dict(relation='N0_dominates',lambda_refresh=None,favors='none, except possible boundary tie')
    f=Fraction(dh,-dt)
    return dict(relation='tradeoff',lambda_refresh=fraction_record(f),favors='above' if dh>0 else 'below')

def analyze(cells):
    lookup={(group(c),c['policy']):c for c in cells}
    labels={k:[] for k in ('STALE_REPLICA_CONFIRMED','ORACLE_REFRESH_HEADROOM','ONLINE_REFRESH_PLAUSIBLE','CACHE_RELIEF_PRESERVED')}
    comparisons=[];prices=[];frontiers=[];life=[]
    for c in cells:
        m=meta(c);baseline=lookup[group(c),'N0'];oracle=lookup[group(c),'OR']
        for scope in ('full','decode'):
            a=c[scope];base=baseline[scope];o=oracle[scope]
            life.append(dict(**m,scope=scope,**{k:v for k,v in a.items() if k.startswith(('replica_','refresh_','victim_','duplicates_never_')) or k=='realized_downstream_peer_saved'}))
            if c['policy']=='N0':continue
            dh=a[H]-base[H];dt=a[T]-base[T];assert dh==a['delta_H2D_bytes'] and dt==a['delta_peer_bytes']
            saved=base[T]-a[T];oracle_saved=base[T]-o[T]
            row=dict(**m,scope=scope,delta_H2D=dh,delta_peer=dt,delta_reload=a['reload_fetches']-base['reload_fetches'],delta_local_service=a['local_service_fraction']-base['local_service_fraction'],delta_unique_coverage=a['mean_unique_resident_experts']-base['mean_unique_resident_experts'],peer_reduction_fraction=ratio(saved,base[T]),h2d_ratio=ratio(a[H],base[H]),unique_coverage_ratio=ratio(a['mean_unique_resident_experts'],base['mean_unique_resident_experts']),pareto_dominates_N0=dominates(a,base),oracle_peer_gain_capture_fraction=ratio(saved,oracle_saved) if oracle_saved>0 else None,OR_has_positive_peer_gain=oracle_saved>0,h2d_ratio_to_OR=ratio(a[H],o[H]),immediate_peer_saved=a['refresh_immediate_peer_saved'],realized_downstream_peer_saved=a['realized_downstream_peer_saved'],**price_case(dh,dt))
            comparisons.append(row)
            assert saved==row['immediate_peer_saved']+row['realized_downstream_peer_saved']
            for lam in PRICES:prices.append(dict(**m,scope=scope,lambda_byte=lam,N0_J_byte=base[H]+lam*base[T],refresh_J_byte=a[H]+lam*a[T],delta_J_byte=dh+lam*dt,refresh_better=dh+lam*dt<0))
            if scope!='decode':continue
            unique_total=round(a['mean_unique_resident_experts']*a['events']);base_unique=round(base['mean_unique_resident_experts']*base['events'])
            if c['policy'] in ('C1','C2'):
                if 5*a[T]<=4*base[T] and 10*a[H]<=11*base[H] and 100*unique_total>=99*base_unique:labels['STALE_REPLICA_CONFIRMED'].append(c['id'])
                if oracle_saved>0 and 5*saved>=3*oracle_saved and 20*a[H]<=21*o[H]:labels['ONLINE_REFRESH_PLAUSIBLE'].append(c['id'])
            elif (4*a[T]<=3*base[T] and 20*a[H]<=23*base[H]) or dominates(a,base):labels['ORACLE_REFRESH_HEADROOM'].append(c['id'])
            if c['cache_ratio']==.6:
                hist=json.loads((OLDROOT/'cells'/f"B{c['batch']}_c30_{c['eviction']}_s{int(c['substitution'])}_rho{c['rho']}.json").read_text())['decode']
                if 2*a[H]<=hist[H] and a[T]<base[T]:labels['CACHE_RELIEF_PRESERVED'].append(c['id'])
    if not any(labels[k] for k in ('STALE_REPLICA_CONFIRMED','ORACLE_REFRESH_HEADROOM','ONLINE_REFRESH_PLAUSIBLE')):labels['NO_REFRESH_HEADROOM']=True
    for g in sorted({group(c) for c in cells}):
        rows=[lookup[g,p] for p in POLICIES]
        for scope in ('decode','full'):
            frontier=[c['policy'] for c in rows if not any(dominates(x[scope],c[scope]) for x in rows)]
            frontiers.append(dict(batch=g[0],cache_ratio=g[1],eviction=g[2],substitution=g[3],rho=g[4],scope=scope,pareto_policies=frontier,envelope=serial_envelope([(c['policy'],c[scope][H],c[scope][T]) for c in rows])))
    table('grid_points',[flat(c) for c in cells]);table('matched_comparisons',comparisons);table('resource_prices',prices);table('frontier_summary',frontiers);table('refresh_lifecycle',life);write('interpretation.json',labels)
    return comparisons,frontiers,labels


def ages(cells):
    # Read future demand only after every cell has finished; never a C1/C2 input.
    lookups={}
    for batch in (8,32):
        raw=json.loads((OLDROOT/f'captures/B{batch}/rank0.json').read_text());index={}
        for item in raw['events']:
            if item['step']==0:continue
            selected=np.asarray(item['raw_selected_experts']);origins=np.asarray(item['origin_ranks'])
            for rank in range(4):
                for expert in np.unique(selected[origins==rank]):index.setdefault((item['layer'],int(expert),rank),[]).append(item['event'])
        lookups[batch]=index;del raw
    rows=[];histograms=[]
    for c in cells:
        swaps=json.loads((ROOT/'swaps'/(c['id']+'.json')).read_text());index=lookups[c['batch']]
        for scope,start in (('full',0),('decode',48)):
            selected=[r for r in swaps if r['event']>=start]
            assert len(selected)==c[scope]['refresh_admissions']
            exposed=dead=no_future=0
            for r in selected:
                future=index.get((r['victim_layer'],r['victim_expert'],r['rank']),[])
                has_opportunity=8*48+r['victim_layer']>r['event']
                if has_opportunity:
                    exposed+=1;dead+=bisect.bisect_right(future,r['event'])==len(future)
                else:no_future+=1
            out=dict(**meta(c),scope=scope,refreshes=len(selected),victims_with_future_decode_opportunity=exposed,victims_with_no_later_raw_rank_demand=dead,raw_demand_dead_fraction=dead/exposed if exposed else None,no_future_opportunity=no_future)
            for metric in ('victim_copy_age','victim_idle_age'):
                values=[r[metric] for r in selected];counts=np.histogram(values,bins=BINS)[0]
                assert int(counts.sum())==len(selected)
                out[metric+'_p50']=float(np.quantile(values,.5)) if values else None;out[metric+'_p90']=float(np.quantile(values,.9)) if values else None
                for lo,hi,count in zip(BINS[:-1],BINS[1:],counts):histograms.append(dict(**meta(c),scope=scope,metric=metric,lower_inclusive=lo,upper_exclusive=hi,count=int(count)))
            rows.append(out)
    table('stale_age_summary',rows);table('stale_age_histograms',histograms)
    return rows,histograms

def figures(cells,comparisons,age_rows,histograms):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    out=P/'figures';out.mkdir(exist_ok=True)
    colors={'N0':'#555555','C1':'#2674b7','C2':'#51a5df','O4':'#d58622','OR':'#b94741'}
    markers={'N0':'s','C1':'o','C2':'^','O4':'D','OR':'*'};saved=[]
    def save(fig,name):
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/(name+'.png'),dpi=145);plt.close(fig);saved.append(name+'.png')
    for rho in (.125,.25):
        fig,axes=plt.subplots(2,4,figsize=(15,7))
        for i,cache in enumerate((.4,.6)):
            for j,(ev,sub) in enumerate((('lru',False),('lru',True),('gate',False),('gate',True))):
                ax=axes[i,j]
                for policy in POLICIES:
                    c=next(c for c in cells if (c['batch'],c['cache_ratio'],c['eviction'],c['substitution'],c['rho'],c['policy'])==(32,cache,ev,sub,rho,policy));a=c['decode']
                    ax.scatter(a[T]/2**20,a[H]/2**30,label=policy,color=colors[policy],marker=markers[policy],s=45)
                ax.set(title=f'cache{cache:.0%} {ev} sub={sub}',xlabel='Peer MiB',ylabel='H2D GiB');ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7);fig.suptitle(f'B32 decode resource points, rho={rho}; both lower is better')
        save(fig,'B32_frontiers_rho'+str(rho))
    decode=[r for r in comparisons if r['scope']=='decode']
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(batch,cache) in zip(axes,((8,.6),(32,.4),(32,.6))):
        for i,policy in enumerate(POLICIES[1:]):
            vals=[100*r['peer_reduction_fraction'] for r in decode if (r['batch'],r['cache_ratio'],r['policy'])==(batch,cache,policy)]
            med=float(np.median(vals));ax.errorbar(i,med,yerr=[[med-min(vals)],[max(vals)-med]],color=colors[policy],marker=markers[policy],capsize=4)
        ax.axhline(0,color='gray',linewidth=.7);ax.set_xticks(range(4),POLICIES[1:]);ax.set(title=f'B{batch} cache{cache:.0%}',ylabel='Peer reduction vs N0 (%)');ax.grid(alpha=.2)
    fig.suptitle('Matched-profile median and min/max (equal weight per profile)');save(fig,'batch_cache_benefit')
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for i,ev in enumerate(('lru','gate')):
        for j,sub in enumerate((False,True)):
            ax=axes[i,j]
            for policy in POLICIES[1:]:
                for batch in (8,32):
                    rr=[r for r in decode if (r['eviction'],r['substitution'],r['policy'],r['batch'])==(ev,sub,policy,batch)]
                    ax.scatter([100*(r['h2d_ratio']-1) for r in rr],[100*r['peer_reduction_fraction'] for r in rr],color=colors[policy],marker=markers[policy],facecolors=colors[policy] if batch==32 else 'none',label=f'{policy} B{batch}',s=35)
            ax.axhline(0,color='gray');ax.axvline(0,color='gray');ax.set(title=f'{ev}, substitution {sub}',xlabel='H2D change vs N0 (%)',ylabel='Peer reduction vs N0 (%)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=6,ncol=2);fig.suptitle('Eviction and substitution effects; upper-left is favorable');save(fig,'policy_tradeoffs')
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for i,batch in enumerate((8,32)):
        for j,metric in enumerate(('victim_copy_age','victim_idle_age')):
            ax=axes[i,j]
            for policy in POLICIES[1:]:
                counts=[sum(r['count'] for r in histograms if (r['batch'],r['scope'],r['policy'],r['metric'],r['lower_inclusive'])==(batch,'decode',policy,metric,lo)) for lo in BINS[:-1]]
                total=sum(counts)
                if total:ax.step(list(range(len(counts))),np.cumsum(counts)/total,where='post',color=colors[policy],label=policy)
            ax.set_xticks(range(len(BINS)-1),['1','8','16','32','48','96','192','end']);ax.set(title=f'B{batch} {metric}',xlabel='Age bin upper edge (layer events)',ylabel='Pooled cumulative fraction',ylim=(0,1.02));ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Age of refreshed victims, decode; weighted by observed swaps');save(fig,'stale_age')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,batch in zip(axes,(8,32)):
        for policy in ('C1','C2'):
            rr=[r for r in decode if (r['batch'],r['policy'])==(batch,policy) and r['OR_has_positive_peer_gain']]
            if not rr:continue
            # Obtain OR reductions directly to handle a zero current-policy gain.
            ors={(r['cache_ratio'],r['eviction'],r['substitution'],r['rho']):r for r in decode if r['batch']==batch and r['policy']=='OR'}
            x=[100*ors[r['cache_ratio'],r['eviction'],r['substitution'],r['rho']]['peer_reduction_fraction'] for r in rr]
            ax.scatter(x,[100*r['peer_reduction_fraction'] for r in rr],color=colors[policy],marker=markers[policy],label=policy)
        limits=ax.get_xlim();upper=max(1,limits[1]);ax.plot([0,upper],[0,.6*upper],color='gray',linestyle='--',label='60% of OR gain')
        ax.set(title=f'B{batch}: OR-positive profiles',xlabel='OR peer reduction (%)',ylabel='Current-only peer reduction (%)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Current-demand capture of oracle benefit; H2D qualification is in tables');save(fig,'online_vs_oracle')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,batch in zip(axes,(8,32)):
        for i,policy in enumerate(POLICIES[1:]):
            rr=[r for r in age_rows if (r['batch'],r['policy'],r['scope'])==(batch,policy,'decode')]
            denom=sum(r['victims_with_future_decode_opportunity'] for r in rr);num=sum(r['victims_with_no_later_raw_rank_demand'] for r in rr)
            ax.bar(i,num/denom if denom else 0,color=colors[policy]);ax.text(i,num/denom if denom else 0,f'{num}/{denom}',ha='center',va='bottom',fontsize=7)
        ax.set_xticks(range(4),POLICIES[1:]);ax.set(title=f'B{batch}',ylabel='No later raw rank demand fraction',ylim=(0,1.1));ax.grid(axis='y',alpha=.2)
    fig.suptitle('Victim raw-demand diagnostic; exclude victims with no remaining decode opportunity');save(fig,'victim_future_demand')
    return saved

def report(cells,comparisons,labels,validation):
    lookup={(group(c),c['policy']):c for c in cells};decode=[r for r in comparisons if r['scope']=='decode']
    passed=[k for k,v in labels.items() if v]
    current=[r for r in decode if r['policy'] in ('C1','C2')]
    best=max(current,key=lambda r:r['peer_reduction_fraction'])
    oracle=[r for r in decode if r['policy'] in ('O4','OR')];best_oracle=max(oracle,key=lambda r:r['peer_reduction_fraction'])
    rows=['# Dynamic stale-replica refresh: bounded results','',
          '**Complete: '+', '.join(passed)+'**.','',
          f"All 120 CPU cells passed (80 B32, 40 B8). The 24 N0 cells exactly reproduce the d666414 full/decode counters and final state hashes and are reused in this matrix. No GPU/model capture or timing run was made.",'',
          f"Largest observed current-only peer reduction: {100*best['peer_reduction_fraction']:.2f}% at `{best['id']}`, with H2D change {100*(best['h2d_ratio']-1):+.2f}% and mean unique-coverage change {100*(best['unique_coverage_ratio']-1):+.2f}%. "
          f"Largest oracle peer reduction: {100*best_oracle['peer_reduction_fraction']:.2f}% at `{best_oracle['id']}`, with H2D change {100*(best_oracle['h2d_ratio']-1):+.2f}%. These extrema do not themselves override the predeclared joint thresholds.",'',
          'This is frozen-route byte accounting over one prefill and eight decode forwards. Substitution accuracy, autoregressive route stability, latency and bandwidth equivalence were not evaluated.','',
          '## Predeclared labels','']
    for label,witnesses in labels.items():
        rows.append(f'- **{label}**: '+(str(len(witnesses))+' qualifying matched cells.' if isinstance(witnesses,list) else 'No current/oracle headroom label passed.'))
    rows+=['','All witnesses are listed in interpretation.json. ONLINE_REFRESH_PLAUSIBLE requires strictly positive OR peer savings; a zero or negative OR gain cannot create a vacuous pass. CACHE_RELIEF_PRESERVED compares cache60 against same-batch historical cache30, with identical eviction/substitution/rho. These labels are existence claims, not universal policy improvements.','',
           '## Decode matched resource changes','',
           'Each policy entry is **peer reduction % / H2D change %** relative to its matched N0. Positive peer reduction is favorable; negative H2D change is favorable.','',
           '| Batch | Cache | Eviction | Sub | rho | N0 H2D GiB / peer MiB | C1 | C2 | O4 | OR |',
           '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for g in sorted({group(c) for c in cells},key=lambda g:(-g[0],*g[1:])):
        base=lookup[g,'N0']['decode'];parts=[]
        for policy in POLICIES[1:]:
            a=lookup[g,policy]['decode'];parts.append(f"{100*(base[T]-a[T])/base[T]:+.2f}% / {100*(a[H]-base[H])/base[H]:+.2f}%")
        rows.append(f'| B{g[0]} | {g[1]:.0%} | {g[2]} | {"ON" if g[3] else "OFF"} | {g[4]} | {base[H]/2**30:.3f} / {base[T]/2**20:.3f} | '+' | '.join(parts)+' |')
    rows+=['','## Batch and cache comparison','',
           'Equal-weight profile medians and ranges below summarize the eight eviction/substitution/rho combinations in each batch/cache group; they are descriptive and are not confidence intervals. The matched table above retains the LRU/GATE and substitution interactions.','',
           '| Batch/cache | Policy | Peer reduction median [min, max] % | H2D change median % | Pareto improvements / 8 |',
           '|---|---|---:|---:|---:|']
    for batch,cache in ((32,.4),(32,.6),(8,.6)):
        for policy in POLICIES[1:]:
            rr=[r for r in decode if (r['batch'],r['cache_ratio'],r['policy'])==(batch,cache,policy)]
            peer=[100*r['peer_reduction_fraction'] for r in rr];h2d=[100*(r['h2d_ratio']-1) for r in rr]
            rows.append(f'| B{batch}/cache{cache:.0%} | {policy} | {np.median(peer):.2f} [{min(peer):.2f}, {max(peer):.2f}] | {np.median(h2d):+.2f} | {sum(r["pareto_dominates_N0"] for r in rr)} |')
    age_rows=json.loads((P/'stale_age_summary.json').read_text())
    rows+=['','## Observed victim demand','',
           'Counts pool decode swaps across each batch. No later raw demand is measured on the removed copy\'s rank, conditional on remaining same-layer decode opportunity. It does not imply no remote service or permanent death. Each cell\'s copy-age/idle-age quantiles and the pooled age distributions are supplied separately.','',
           '| Batch | Policy | No later raw rank demand / exposed victims | No remaining opportunity | Immediate savings MiB | Downstream residual MiB |',
           '|---|---|---:|---:|---:|---:|']
    for batch in (32,8):
        for policy in POLICIES[1:]:
            aa=[r for r in age_rows if (r['batch'],r['policy'],r['scope'])==(batch,policy,'decode')]
            rr=[r for r in decode if (r['batch'],r['policy'])==(batch,policy)]
            dead=sum(r['victims_with_no_later_raw_rank_demand'] for r in aa);exposed=sum(r['victims_with_future_decode_opportunity'] for r in aa)
            tail=sum(r['no_future_opportunity'] for r in aa)
            immediate=sum(r['immediate_peer_saved'] for r in rr)/2**20;downstream=sum(r['realized_downstream_peer_saved'] for r in rr)/2**20
            rows.append(f'| B{batch} | {policy} | {dead}/{exposed} ({100*dead/exposed:.1f}%) | {tail} | {immediate:+.2f} | {downstream:+.2f} |' if exposed else f'| B{batch} | {policy} | 0/0 (undefined) | {tail} | {immediate:+.2f} | {downstream:+.2f} |')
    rows+=['','## Mechanism and attribution','',
           'Every refresh preserves the global unique-key set and duplicate count at the swap instant. Later eviction, reload, primary-owner and substitution trajectories can diverge, so mean unique coverage is measured rather than assumed equal. Refresh fetches are included in replica and total H2D counts.','',
           'Immediate savings are exact before/after service-byte differences at the accepted swap. The realized downstream residual equals total N0-minus-refresh peer savings minus those immediate savings. It measures effects of preceding refresh decisions on the subsequent real trajectory, including admissions and substitution; it is not additive causal credit for individual swaps. Both values, including negative residuals, remain in matched_comparisons.csv.','',
           'Victim copy age and idle age are measured in global layer events. Lifecycle summaries retain right-censored observed lower bounds and decode-born replica cohorts. Never-reused means no service in a later event before eviction/end; service at admission is excluded. A replica can become primary without ending its physical lifetime.','',
           'The post-run victim-demand diagnostic uses raw rank demand only and is never available to C1/C2. It reports no later observed demand only among victims with at least one remaining same-layer decode opportunity; end-of-trace cases with no remaining opportunity are counted separately. This does not assert death beyond the short trace, equivalent effective demand under substitution, or absence of remote service by a primary copy.','',
           '## Oracle interpretation','',
           'O4/OR score a static present-cache counterfactual over current service and four/all future decode opportunities for both affected layers. They include exact dispatch coalescing and duplicate-primary promotion. Future global misses use a virtual deterministic first-copy destination equally in both worlds. No future eviction or cache policy trajectory is predicted. Effective future routes use the unchanged substitution policy against the present global unique set. These are future-demand diagnostics, not guaranteed optimal sequential controllers; C1/C2 can exceed their realized benefit. See EXECUTION.md.','',
           '## Resource prices','',
           'matched_comparisons.json stores exact rational N0-versus-policy break-even prices and the favorable side of each crossing. Dominance and equal/same-direction changes are labeled separately. resource_prices.csv evaluates only lambda=32,64,96,128,256. frontier_summary.json contains the exact five-policy lower envelope and raw H2D-peer nondominated policies. Lambda is a byte-resource price, not a hardware slowdown or latency ratio.','',
           '## Validation and resources','',
           f"- All 120 cells completed under one CPU replay process, CUDA hidden, BLAS/OMP=1, and an 8-GiB address-space limit. Peak replay RSS: {validation['peak_cpu_rss_mib']:.2f} MiB.",
           '- 24 N0 references match every prior full/decode metric and final cache hash.',
           '- Each refresh checks inactive victim, surviving global copy, unchanged unique coverage/duplicate count, slot/capacity safety, and exact immediate traffic reduction.',
           '- All event traffic checks send/receive transpose parity and legal resident destinations.',
           '- Independent tiny fixtures compare chosen swaps against enumerated complete before/after traffic, including same-layer coalescing, substitution and R4 primary promotion.',
           '- Source traces, similarity, baseline receipts and historical replay code are hash-verified against d666414.',
           '- No model execution, new capture, quality evaluation, E2E/NCCL timing, larger batch or additional sweep. GPU jobs are untouched. Stop for owner review.','',
           '## Outputs','',
           'grid_points: all full/decode counters. matched_comparisons and resource_prices: exact paired resource changes. refresh_lifecycle: reuse, lifetime and refresh accounting. stale_age_summary/histograms: victim ages and post-run raw-demand evidence. frontier_summary: matched Pareto points and exact envelopes. validation.json and source_verification.json: correctness/provenance. Cell and individual swap receipts stay outside git and are referenced by hashes.','',
           '## Figures','']
    for name in validation['figures']:rows+=['!['+name+'](figures/'+name+')','']
    (P/'RESULTS.md').write_text('\n'.join(rows).rstrip()+'\n')

def main():
    stages=[json.loads((P/name).read_text()) for name in ('N0_validation.json','refresh_progress.json')]
    assert all(s['status']=='PASS' for s in stages) and [len(s['cells']) for s in stages]==[24,96]
    for s in stages:
        for path,expected in s['source_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
    cells=[];receipts=[]
    for stage in stages:
        for row in stage['cells']:
            file=ROOT/'cells'/(row['id']+'.json');assert hashlib.sha256(file.read_bytes()).hexdigest()==row['sha256']
            cell=json.loads(file.read_text());cells.append(cell)
            swap=ROOT/'swaps'/(cell['id']+'.json')
            receipts.append(dict(id=cell['id'],cell_path=str(file),cell_sha256=row['sha256'],swap_path=str(swap),swap_sha256=hashlib.sha256(swap.read_bytes()).hexdigest()))
    assert len(cells)==len({c['id'] for c in cells})==120
    assert Counter(c['batch'] for c in cells)=={32:80,8:40} and all(n==24 for n in Counter(c['policy'] for c in cells).values())
    for c in cells:
        for scope,n in (('full',432),('decode',384)):
            a=c[scope];assert a['events']==n and a['peak_duplicate_slots']<=c['duplicate_cap']
            assert a[H]==a['total_fetches']*9*2**20
            assert a['total_fetches']==a['first_copy_fetches']+a['reload_fetches']+a['replica_fetches']
            assert a[T]==4096*(a['remote_token_rank_pairs']+a['remote_expert_routes'])
            assert a['refresh_admissions']==a['refresh_duplicate_evictions']
    comparisons,frontiers,labels=analyze(cells)
    age_rows,histograms=ages(cells);saved=figures(cells,comparisons,age_rows,histograms)
    rows_checks={}
    for name,count in [('grid_points',120),('matched_comparisons',192),('resource_prices',960),('refresh_lifecycle',240),('frontier_summary',48),('stale_age_summary',240),('stale_age_histograms',3840)]:
        data=json.loads((P/(name+'.json')).read_text());csvdata=list(csv.DictReader((P/(name+'.csv')).open()));assert len(data)==len(csvdata)==count
        for a,b in zip(data,csvdata):
            for k,v in a.items():
                if isinstance(v,(list,dict)):assert json.loads(b[k])==v
                else:assert b[k]==('' if v is None else str(v))
        rows_checks[name]=count
    validation=dict(status='PASS',cells=120,B32=80,B8=40,N0_byte_counter_and_state_parity=24,source_hashes=stages[-1]['source_hashes'],peak_cpu_rss_mib=max(s['peak_rss_mib'] for s in stages),address_space_limit_bytes=8*2**30,cpu_replay_processes=1,threads=1,cuda_visible_devices='',new_model_runs=0,new_gpu_runs=0,labels=[k for k,v in labels.items() if v],csv_json_equal_row_counts=rows_checks,figures=saved,cell_seconds=sum(c['seconds'] for c in cells),refresh_swaps=sum(c['full']['refresh_admissions'] for c in cells),all_unique_set_and_duplicate_count_swap_checks=True,all_current_only_policies_have_no_future_provider=True,policy_tests=json.loads((P/'policy_tests.json').read_text()),no_quality_or_timing_claim=True)
    write('validation.json',validation);write('cell_receipts.json',receipts);report(cells,comparisons,labels,validation)
    print(json.dumps({k:validation[k] for k in ('status','cells','labels','peak_cpu_rss_mib','cell_seconds','refresh_swaps')}))
if __name__=='__main__':main()
