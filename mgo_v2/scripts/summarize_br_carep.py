#!/usr/bin/env python3
"""Validated two-horizon resource findings; no accuracy or timing claims."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path('/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003')
P=Path(__file__).resolve().parents[1]/'experiments/br_ca_carep_cpu_headroom_20261003'
META=('id','dataset','horizon','world','batch','cache','eviction','substitution','policy','seed','seed_audit')
H='H2D_bytes';T='peer_bytes'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
def write(name,obj):(P/name).write_text(json.dumps(obj,separators=(',',':'),allow_nan=False)+'\n')
def table(name,rows):
    write(name+'.json',rows);fields=list(dict.fromkeys(k for r in rows for k in r))
    with (P/(name+'.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def meta(c):return {k:c[k] for k in META}
def base(c):return tuple(c[k] for k in ('dataset','horizon','world','batch','cache','eviction','substitution'))
def ratio(a,b):return a/b if b else None
def reduction(a,b):return (b-a)/b if b else None
def compare(c,b,kind):
    a=c['decode'];d=b['decode']
    return dict(**meta(c),reference_id=b['id'],kind=kind,peer_reduction_fraction=reduction(a[T],d[T]),H2D_ratio=ratio(a[H],d[H]),delta_H2D=a[H]-d[H],delta_peer=a[T]-d[T],delta_reload_bytes=a['reload_bytes']-d['reload_bytes'],delta_local_service=a['local_service_fraction']-d['local_service_fraction'],delta_unique_coverage=a['mean_unique_resident_experts']-d['mean_unique_resident_experts'],reference_H2D=d[H],reference_peer=d[T],H2D=a[H],peer=a[T],rank_fetch_imbalance=a['max_event_rank_fetch_imbalance'],reference_rank_fetch_imbalance=d['max_event_rank_fetch_imbalance'],pareto_dominates=a[H]<=d[H] and a[T]<=d[T] and (a[H]<d[H] or a[T]<d[T]),replica_admissions=a['replica_fetches'],replicas_reused=a['replicas_reused'],replica_victim_reloads=a['replica_victim_reloads'])
def analyze(cells):
    main=[c for c in cells if not c['seed_audit']];lookup={(base(c),c['policy']):c for c in main};grid=[];comparisons=[];sub=[];seed=[]
    labels={k:[] for k in ('CA_HEADROOM','CA_STRONG_HEADROOM','CA_REP_HEADROOM','CA_REP_TRADEOFF','SUBSTITUTION_RELIEVES_MISS_PRESSURE','HIGH_MISS_PRESSURE')}
    for c in cells:
        for scope in ('decode','full'):
            metrics=dict(c[scope])
            for key in ('resident_copies','unique_resident_experts','duplicate_copies'):metrics[key+'_event_sum']=metrics.pop(key)
            grid.append(dict(**meta(c),scope=scope,global_slots=c['global_slots'],**metrics))
        a=c['decode']
        if c['seed_audit']:
            b=lookup[base(c),'BR'];seed.append(compare(c,b,'BR_seed'));continue
        if a['residual_miss_routes']*10>=3*a['raw_routes'] or a['evictions']>=c['global_slots']*c['horizon']:labels['HIGH_MISS_PRESSURE'].append(c['id'])
        if c['policy']=='CA':
            b=lookup[base(c),'BR'];d=b['decode'];comparisons.append(compare(c,b,'BR_to_CA'))
            constraints=98*d[H]<=100*a[H]<=102*d[H] and a['max_event_rank_fetch_imbalance']<=d['max_event_rank_fetch_imbalance']+1
            if constraints and 10*a[T]<=9*d[T]:labels['CA_HEADROOM'].append(c['id'])
            if constraints and 4*a[T]<=3*d[T]:labels['CA_STRONG_HEADROOM'].append(c['id'])
        elif c['policy']=='CA-rep':
            b=lookup[base(c),'CA'];d=b['decode'];comparisons.append(compare(c,b,'CA_to_CA-rep'))
            if 5*a[T]<=4*d[T]:labels['CA_REP_HEADROOM' if 20*a[H]<=23*d[H] else 'CA_REP_TRADEOFF'].append(c['id'])
        if c['substitution']:
            g=list(base(c));g[-1]=False;b=lookup[tuple(g),c['policy']];d=b['decode'];rr=compare(c,b,'OFF_to_ON')
            rr.update(residual_miss_fraction=a['residual_miss_routes_fraction'],reference_residual_miss_fraction=d['residual_miss_routes_fraction'],substituted_route_fraction=a['substituted_routes_fraction'],substituted_gate_mass_fraction=a['substituted_mass_fraction'],resident_substitute_fraction=a['resident_substitute_hits_fraction'],shared_admission_fraction=a['shared_admission_substitutions_fraction']);sub.append(rr)
            miss_pass=d['residual_miss_routes']>0 and 5*a['residual_miss_routes']*d['raw_routes']<=4*d['residual_miss_routes']*a['raw_routes']
            reload_pass=d['reload_bytes']>0 and 5*a['reload_bytes']<=4*d['reload_bytes']
            if miss_pass or reload_pass:labels['SUBSTITUTION_RELIEVES_MISS_PRESSURE'].append(c['id'])
    table('grid_points',grid);table('policy_comparisons',comparisons);table('substitution_comparisons',sub);table('seed_audit',seed);write('interpretation.json',labels)
    # Matched global request sets, preserving all other axes.
    same=[]
    for c in main:
        if c['world']!=4 or c['batch'] not in (16,32,64):continue
        g=list(base(c));g[2]=8;g[3]=c['batch']//2;b=lookup[tuple(g),c['policy']]
        rr=compare(b,c,'R4_to_R8_same_global');rr['global_requests']=4*c['batch'];same.append(rr)
    table('same_global_comparisons',same)
    return grid,comparisons,sub,seed,same,labels
def validate(cells,progress):
    assert len(cells)==len({c['id'] for c in cells})==1552
    assert sum(not c['seed_audit'] for c in cells)==1536
    main=[c for c in cells if not c['seed_audit']];assert Counter(c['horizon'] for c in main)=={64:768,256:768}
    refs={tuple(c[k] for k in ('dataset','world','batch','cache','eviction','substitution','policy','horizon')):c for c in main}
    pairs=0;replica_free_pairs=0;receipts=[]
    for c in cells:
        assert c['status']=='PASS' and c['source_hashes']==progress['source_hashes']
        assert c['global_slots']==6144*c['cache']//100 and sum(c['per_rank_slots'])==c['global_slots']
        file=Path(c['event_path']);assert sha(file)==c['event_sha256']
        raw=np.load(file);assert raw['rows'].shape==((c['horizon']+1)*48,48)
        assert np.array_equal(raw['rank_fetches'].sum(1),raw['rows'][:,20]+raw['rows'][:,21])
        for scope,n in (('full',48*(c['horizon']+1)),('decode',48*c['horizon'])):
            a=c[scope];assert a['events']==n and a[H]==a['total_fetches']*9437184 and a[T]==4096*(a['remote_token_rank_pairs']+a['remote_expert_routes'])
            assert a['exact_global_hits']+a['resident_substitute_hits']+a['shared_admission_substitutions']+a['residual_miss_routes']==a['raw_routes']
            assert a['local_services']+a['remote_expert_routes']==a['effective_routes'] and a['peak_resident_copies']<=c['global_slots']
            assert abs(a['raw_gate_mass']-a['effective_gate_mass'])<1e-6
            assert abs(a['exact_global_hit_mass']+a['resident_substitute_mass']+a['shared_admission_mass']+a['residual_miss_mass']-a['raw_gate_mass'])<1e-6
            assert a['max_event_rank_fetch_imbalance']<=1
            if c['policy']=='CA-rep':
                assert sum(a[k] for k in ('replica_score_lt_quarter','replica_score_quarter_half','replica_score_half_one','replica_score_one_two','replica_score_ge_two'))==a['unique_miss_expert_events']
                assert a['replica_fetches']+a['replica_blocked_by_pinning']==a['replica_score_one_two']+a['replica_score_ge_two']
        if c['policy'] in ('BR','CA'):assert c['full']['replica_fetches']==0 and c['full']['peak_duplicate_copies']==0
        if not c['seed_audit'] and c['horizon']==256 and c['policy'] in ('BR','CA'):
            key=tuple(c[k] for k in ('dataset','world','batch','cache','eviction','substitution','policy'))+(64,);short=refs[key];a=np.load(short['event_path']);assert np.array_equal(a['rows'],raw['rows'][:65*48]) and np.array_equal(a['rank_fetches'],raw['rank_fetches'][:65*48]);pairs+=1
        if not c['seed_audit'] and c['policy']=='CA-rep' and c['full']['replica_fetches']==0:
            key=tuple(c[k] for k in ('dataset','world','batch','cache','eviction','substitution'))+('CA',c['horizon']);a=refs[key]
            assert c['final_state_sha256']==a['final_state_sha256']
            for metric in (H,T,'reload_bytes','local_services','evictions'):assert c['full'][metric]==a['full'][metric]
            replica_free_pairs+=1
        receipts.append(dict(id=c['id'],path=str(ROOT/'cells'/(c['id']+'.json')),sha256=sha(ROOT/'cells'/(c['id']+'.json')),event_path=str(file),event_sha256=c['event_sha256'],seconds=c['seconds'],peak_rss_bytes=c['peak_rss_bytes']))
    assert pairs==512;write('cell_receipts.json',receipts)
    return dict(status='PASS',cells=1552,main_cells=1536,seed_audit_cells=16,BR_CA_exact_prefix_pairs=pairs,zero_replica_CA_identity_pairs=replica_free_pairs,peak_cell_rss_bytes=max(c['peak_rss_bytes'] for c in cells),peak_aggregate_replay_rss_bytes=progress['peak_aggregate_rss_bytes'],csv_json_checks={},source_hashes=progress['source_hashes'],raw_capture_runs=2,model_loading_sessions=1,new_calibration_runs=0,no_quality_or_timing_claim=True)

def aggregates(cells,comparisons,labels):
    rows=[]
    for dataset in ('MATH','ShareGPT'):
        for horizon in (64,256):
            for kind in ('BR_to_CA','CA_to_CA-rep'):
                rr=[r for r in comparisons if (r['dataset'],r['horizon'],r['kind'])==(dataset,horizon,kind)]
                peer=[100*r['peer_reduction_fraction'] for r in rr if r['peer_reduction_fraction'] is not None];h2d=[100*(r['H2D_ratio']-1) for r in rr if r['H2D_ratio'] is not None]
                ids={r['id'] for r in rr};row=dict(dataset=dataset,horizon=horizon,comparison=kind,cells=len(rr),peer_reduction_percent_median=float(np.median(peer)),peer_reduction_percent_min=min(peer),peer_reduction_percent_max=max(peer),H2D_change_percent_median=float(np.median(h2d)),H2D_change_percent_min=min(h2d),H2D_change_percent_max=max(h2d),pareto_improvements=sum(r['pareto_dominates'] for r in rr))
                for label in ('CA_HEADROOM','CA_STRONG_HEADROOM','CA_REP_HEADROOM','CA_REP_TRADEOFF'):row[label]=len(ids&set(labels[label]))
                rows.append(row)
    table('group_summary',rows)
    return rows

def figures(cells,comp,sub,same):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    P.joinpath('figures').mkdir(exist_ok=True);saved=[];main=[c for c in cells if not c['seed_audit']]
    colors={8:'#3979b9',16:'#4aa174',32:'#d79932',64:'#b35c74'}
    def save(fig,name):
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(P/'figures'/(name+'.png'),dpi=140);plt.close(fig);saved.append(name+'.png')
    panels=[(d,h) for d in ('MATH','ShareGPT') for h in (64,256)]
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,(d,h) in zip(axes.ravel(),panels):
        values=np.array([[np.median([100*r['peer_reduction_fraction'] for r in comp if (r['dataset'],r['horizon'],r['kind'],r['world'],r['batch'])==(d,h,'BR_to_CA',R,B)]) for B in (8,16,32,64)] for R in (4,8)])
        vmax=max(1,float(np.max(np.abs(values))));im=ax.imshow(values,cmap='RdBu',vmin=-vmax,vmax=vmax,aspect='auto')
        for i in range(2):
            for j in range(4):ax.text(j,i,f'{values[i,j]:.1f}%',ha='center',va='center',color='black',bbox=dict(facecolor='white',alpha=.7,edgecolor='none'))
        ax.set_xticks(range(4),(8,16,32,64));ax.set_yticks(range(2),('R4','R8'));ax.set(title=f'{d}, decode{h}',xlabel='Local batch');fig.colorbar(im,ax=ax,label='Peer reduction (%)')
    fig.suptitle('BR → CA: median matched peer reduction over cache/eviction/substitution');save(fig,'placement_R_batch')
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,(d,h) in zip(axes.ravel(),panels):
        for B in (8,16,32,64):
            for R,marker in ((4,'o'),(8,'^')):
                rr=[r for r in comp if (r['dataset'],r['horizon'],r['kind'],r['batch'],r['world'])==(d,h,'CA_to_CA-rep',B,R) and r['H2D_ratio'] is not None]
                ax.scatter([100*(r['H2D_ratio']-1) for r in rr],[100*r['peer_reduction_fraction'] for r in rr],color=colors[B],marker=marker,s=20,alpha=.65,label=f'B{B}/R{R}')
        ax.axhline(20,color='gray',linestyle='--',linewidth=.7);ax.axvline(15,color='gray',linestyle='--',linewidth=.7);ax.set(title=f'{d}, decode{h}',xlabel='H2D change vs CA (%)',ylabel='Peer reduction vs CA (%)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=6,ncol=2);fig.suptitle('CA → CA-rep: realized resources; upper-left is favorable');save(fig,'replication_tradeoffs')
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,(d,h) in zip(axes.ravel(),panels):
        for B in (8,16,32,64):
            for on,style in ((False,'-'),(True,'--')):
                values=[np.mean([100*c['decode']['residual_miss_routes_fraction'] for c in main if (c['dataset'],c['horizon'],c['policy'],c['batch'],c['cache'],c['substitution'])==(d,h,'BR',B,cache,on)]) for cache in (30,40,50,60)]
                ax.plot((30,40,50,60),values,style,marker='o',color=colors[B],label=f'B{B} sub {on}')
        ax.set(title=f'{d}, decode{h}',xlabel='Global cache (%)',ylabel='Residual source-miss routes (%)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=6,ncol=2);fig.suptitle('BR miss pressure; mean over R and eviction; dashed = substitution ON');save(fig,'miss_pressure')
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    pcolors={'BR':'#555555','CA':'#3274a1','CA-rep':'#bd763b'}
    for ax,(d,h) in zip(axes.ravel(),panels):
        for policy in pcolors:
            rr=[c for c in main if (c['dataset'],c['horizon'],c['policy'])==(d,h,policy)]
            ax.scatter([c['decode']['cache_turnover_per_decode_step'] for c in rr],[c['decode']['reload_bytes']/2**30 for c in rr],s=15,alpha=.55,color=pcolors[policy],label=policy)
        ax.set(title=f'{d}, decode{h}',xlabel='Evictions / global slots / decode step',ylabel='Decode reload GiB');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Physical cache turnover and expert reload pressure');save(fig,'turnover_reload')
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,(d,h) in zip(axes.ravel(),panels):
        for j,policy in enumerate(pcolors):
            for i,n in enumerate((64,128,256)):
                rr=[100*r['peer_reduction_fraction'] for r in same if (r['dataset'],r['horizon'],r['policy'],r['global_requests'])==(d,h,policy,n)];med=float(np.median(rr));ax.errorbar(i+(j-1)*.12,med,yerr=[[med-min(rr)],[max(rr)-med]],marker='o',capsize=3,color=pcolors[policy],label=policy if i==0 else None)
        ax.set_xticks(range(3),(64,128,256));ax.axhline(0,color='gray',linewidth=.7);ax.set(title=f'{d}, decode{h}',xlabel='Same global request count',ylabel='Peer reduction R4 → R8 (%)');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Same-global comparison: matched median and min/max, not confidence intervals');save(fig,'same_global_R4_R8')
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,(d,h) in zip(axes.ravel(),panels):
        for j,ev in enumerate(('lru','gate')):
            for i,on in enumerate((False,True)):
                vals=[100*r['peer_reduction_fraction'] for r in comp if (r['dataset'],r['horizon'],r['kind'],r['eviction'],r['substitution'])==(d,h,'BR_to_CA',ev,on)];med=float(np.median(vals));ax.errorbar(i+(j-.5)*.15,med,yerr=[[med-min(vals)],[max(vals)-med]],marker='o',capsize=4,color=('#3779ac','#c08034')[j],label=ev if i==0 else None)
        ax.set_xticks((0,1),('Sub OFF','Sub ON'));ax.set(title=f'{d}, decode{h}',ylabel='BR → CA peer reduction (%)');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Eviction/substitution interaction: matched-profile median and min/max');save(fig,'eviction_substitution')
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    hitkeys=('exact_global_hits_fraction','resident_substitute_hits_fraction','shared_admission_substitutions_fraction','residual_miss_routes_fraction');labels=('Exact global','Resident substitute','Shared new admission','Residual miss');cs=('#3c8d75','#81bc72','#d8b054','#b85c5c')
    for ax,(d,h) in zip(axes.ravel(),panels):
        bottom=np.zeros(4)
        for key,label,color in zip(hitkeys,labels,cs):
            values=np.array([np.mean([100*c['decode'][key] for c in main if (c['dataset'],c['horizon'],c['policy'],c['cache'],c['substitution'])==(d,h,'BR',cache,True)]) for cache in (30,40,50,60)])
            ax.bar(range(4),values,bottom=bottom,color=color,label=label);bottom+=values
        ax.set_xticks(range(4),(30,40,50,60));ax.set(title=f'{d}, decode{h}',xlabel='Global cache (%)',ylabel='Raw route fraction (%)',ylim=(0,100))
    axes[0,0].legend(fontsize=7);fig.suptitle('Disjoint hit/miss accounting: BR/substitution ON, mean across R/B/eviction');save(fig,'hit_partition')
    return saved

def report(cells,comp,sub,seed,same,labels,groups,validation):
    ca=[g for g in groups if g['comparison']=='BR_to_CA'];rep=[g for g in groups if g['comparison']=='CA_to_CA-rep']
    overview=(f"CA reduces median peer bytes by {min(g['peer_reduction_percent_median'] for g in ca):.2f}–{max(g['peer_reduction_percent_median'] for g in ca):.2f}% across the four dataset/horizon groups, with median H2D changes of {min(g['H2D_change_percent_median'] for g in ca):+.2f}–{max(g['H2D_change_percent_median'] for g in ca):+.2f}%. "
              f"Its joint headroom criterion passes in {len(labels['CA_HEADROOM'])}/512 matched comparisons. CA-rep adds median peer reductions of {min(g['peer_reduction_percent_median'] for g in rep):.2f}–{max(g['peer_reduction_percent_median'] for g in rep):.2f}%; its largest observed reduction is {max(g['peer_reduction_percent_max'] for g in rep):.2f}%. "
              'The joint labels below determine whether these changes meet the predeclared thresholds; larger replica gains must be assessed together with their extra H2D.')
    lines=['# BR / CA / CA-rep: two-horizon resource headroom','',
           '**Complete: 1,536 main CPU replays and 16 frozen BR seed-audit replays.**','',
           overview,'',
           'MATH and ShareGPT each have one 512-request exact-model master capture on eight GPUs: one prefill and 256 decode forwards. The same loaded model replicas served both datasets. Decode64 is a prefix of decode256, not a second GPU generation. Existing same-checkpoint FineWeb-Edu 400×128 SERE Frobenius similarity was verified and reused.','',
           'This study measures frozen-route physical resources, not latency, bandwidth equivalence, accuracy or autoregressive quality under intervention. Profile medians/ranges give equal weight to configurations and are not confidence intervals.','',
           '## Predeclared findings','']
    for k,v in labels.items():lines.append(f'- **{k}**: {len(v)} qualifying main cells; witness IDs are in interpretation.json.')
    lines+=['','CA_HEADROOM requires ≥10% peer reduction and H2D within ±2%; CA_STRONG_HEADROOM uses ≥25%, with the same rank-quota check. CA_REP_HEADROOM requires ≥20% peer reduction with H2D increase ≤15%; otherwise a ≥20% reduction is CA_REP_TRADEOFF. Counts below apply these joint conditions, not just the best isolated metric.','',
            '| Dataset | Decode | Comparison | Peer reduction median [min, max] % | H2D change median [min, max] % | Pareto improvements | Joint headroom |',
            '|---|---:|---|---:|---:|---:|---:|']
    for r in groups:
        label='CA_HEADROOM' if r['comparison']=='BR_to_CA' else 'CA_REP_HEADROOM'
        lines.append(f"| {r['dataset']} | {r['horizon']} | {r['comparison']} | {r['peer_reduction_percent_median']:.2f} [{r['peer_reduction_percent_min']:.2f}, {r['peer_reduction_percent_max']:.2f}] | {r['H2D_change_percent_median']:+.2f} [{r['H2D_change_percent_min']:+.2f}, {r['H2D_change_percent_max']:+.2f}] | {r['pareto_improvements']}/128 | {r[label]}/128 |")
    lines+=['','Each group contains the same 128 R/batch/cache/eviction/substitution coordinates. Lower H2D and lower peer are jointly favorable; no single weighted resource winner is imposed.','',
            '## What the longer raw trace adds','',
            '| Dataset | Decode64 mean active experts/layer-event | Decode256 | Top-10% demand share, 64 → 256 | Seen64 pairs recurring later | Early/late hot-set Jaccard |',
            '|---|---:|---:|---:|---:|---:|']
    for name in ('MATH','ShareGPT'):
        a=json.loads((P/(name+'_horizon_audit.json')).read_text());short,long=a['horizons']
        lines.append(f"| {name} | {short['active_experts_mean']:.2f} | {long['active_experts_mean']:.2f} | {100*short['top10_share_mean']:.2f}% → {100*long['top10_share_mean']:.2f}% | {100*a['seen64_pairs_recurring_later_fraction']:.2f}% | {a['mean_hot_set_jaccard']:.3f} |")
    lines+=['','The hot set is the 103 highest-demand expert/rank pairs per layer. Jaccard compares steps 1–64 with 65–256. Recurrence is raw observed demand and does not establish cache residency or quality. The per-layer CSVs also retain event/cumulative demand p50/p90/p99/max, gate mass and recurrence gaps.','',
            '## Cache hits, substitution and miss pressure','',
            'All hit/miss metrics are measured before admission. Exact local hits are a subset of exact global hits. A substituted source can reuse a resident expert or share a newly admitted protected expert; the latter is kept separate from cache hits. Exact global + resident substitute + shared admission + residual source miss partitions raw routes and gate mass exactly. Effective hit = exact global + resident substitute. Effective local hits and post-admission local service are separately recorded.','',
            '| Dataset | Decode | Resident substitute route / mass % | Shared new-admission route / mass % | All substitution route / mass % | Median residual-miss reduction ON vs OFF % |',
            '|---|---:|---:|---:|---:|---:|']
    main=[c for c in cells if not c['seed_audit']]
    for name in ('MATH','ShareGPT'):
        for h in (64,256):
            cc=[c['decode'] for c in main if (c['dataset'],c['horizon'],c['substitution'])==(name,h,True)];rr=[r for r in sub if (r['dataset'],r['horizon'])==(name,h)]
            avg=lambda k:100*float(np.mean([c[k] for c in cc]))
            changes=[100*(1-r['residual_miss_fraction']/r['reference_residual_miss_fraction']) for r in rr if r['reference_residual_miss_fraction']>0]
            lines.append(f"| {name} | {h} | {avg('resident_substitute_hits_fraction'):.2f} / {avg('resident_substitute_mass_fraction'):.2f} | {avg('shared_admission_substitutions_fraction'):.2f} / {avg('shared_admission_mass_fraction'):.2f} | {avg('substituted_routes_fraction'):.2f} / {avg('substituted_mass_fraction'):.2f} | {float(np.median(changes)):.2f} |")
    lines+=['','Hit fractions above are equal-profile means. Full route-count and gate-mass values are in grid_points. HIGH_MISS_PRESSURE means residual source-miss routes ≥30% or at least one global-cache turnover per decode step. Native substitution thresholds remain .20 gate protection and .65 SERE similarity.','',
            '## Placement and replica interpretation','',
            'BR and CA use the identical balanced quota rule for a given current miss count (rank count difference ≤1). Their realized miss sets and numerical quotas can later diverge as cache trajectories evolve, which is why H2D differences are measured. CA exactly maximizes local effective expert-route demand with an integer Hungarian assignment. This minimizes return expert rows for that event, but dispatch coalescing makes it different from globally minimizing dispatch-plus-return bytes. Every reported peer value recomputes both components. Cumulative rank-fetch imbalance is also recorded: balanced event quotas do not imply identical cumulative counts.','',
            'CA-rep only considers newly admitted residual-miss experts. Its V is an optimistic upper bound: two 4096-byte rows times strictly later raw decode demand on one non-primary rank. It admits at V≥9 MiB, at most one replica per new expert, with ordinary unprotected LRU/Gate eviction afterwards. No rho budget, migration, refresh, future eviction or future substitution is used. V bins below/above the threshold are retained for all new experts. This bound can overestimate dispatch savings; realized bytes decide the labels.','',
            'Replica reuse excludes admission-event service. Final never-reused counts include both evicted and still-alive copies without later use. Replica-victim reloads identify a last global disappearance caused directly by replica admission, not a full causal decomposition; matched reload changes versus CA are supplied separately.','',
            'Same-global comparisons use the same first N requests for R4/B16 vs R8/B8, R4/B32 vs R8/B16, and R4/B64 vs R8/B32. Origin mapping, per-rank slots and rank-major GateHistory order change with R. Full-router probabilities are retained so each W128 history is reconstructed for that actual order.','',
            '## BR randomness audit','',
            'Exactly 16 extra cells use seeds 7 and 99: each dataset/horizon has R4/B8/cache30/LRU/OFF and R8/B64/cache60/Gate/ON anchors. Seed42 is already present in the main grid. No seed was selected after observing results.','']
    seed_peer=[100*r['peer_reduction_fraction'] for r in seed if r['peer_reduction_fraction'] is not None];seed_h=[100*(r['H2D_ratio']-1) for r in seed if r['H2D_ratio'] is not None]
    lines.append(f"Across these fixed anchors, alternative-seed peer changes relative to seed42 range from {-max(seed_peer):+.2f}% to {-min(seed_peer):+.2f}%; H2D changes range from {min(seed_h):+.2f}% to {max(seed_h):+.2f}%. The seed audit is limited to these anchors and is not a confidence interval for all cells.")
    lines+=['','## Validation and execution','',
            f"- All 1,552 cells passed. BR/CA have {validation['BR_CA_exact_prefix_pairs']} exact 64-step CPU-prefix matches across horizons.",
            f"- {validation['zero_replica_CA_identity_pairs']} CA-rep cells with zero replicas match CA resources and final state exactly.",
            '- Exact byte identities, raw/effective gate mass, disjoint hit/miss accounting, quota imbalance and physical capacity were checked for every cell.',
            '- Independent tiny dictionary replays, exhaustive assignment optima, R8 fixtures, native substitution and complete before/after traffic checks are preserved in policy_tests.json and pack validation receipts.',
            '- Model weight shards, SERE calibration, source datasets, selected requests and trace files are hash-pinned. No capture was repeated for a CPU configuration.',
            f"- Maximum aggregate replay RSS: {validation['peak_aggregate_replay_rss_bytes']/2**30:.2f} GiB; maximum single-cell RSS: {validation['peak_cell_rss_bytes']/2**20:.1f} MiB. CPU concurrency ≤16, threads=1, CUDA hidden; guards enforce ≤32 GiB aggregate and ≥512 GiB available before each wave.",
            '- Eight resident exact Qwen replicas were loaded once; each processed MATH then ShareGPT. The final decode forward consumes token256 and discards logits, preserving exactly 256 generated tokens and 256 decode route steps. No quality or transport timing was performed.',
            '- The owner explicitly added both CPU horizons after 986ba64. The preceding dynamic-refresh experiment remains stopped at 99/120 cells and was not resumed.','',
            '## Outputs','',
            'grid_points contains full/decode resources and hit/miss accounting for all cells. policy_comparisons, substitution_comparisons and same_global_comparisons preserve matched differences. seed_audit lists all extra-seed comparisons. interpretation contains label witnesses; validation and cell_receipts preserve checks and external raw-result hashes. Horizon audits describe the two raw prefixes.','',
            'Stop for owner review. No Env1/Env2 timing, accuracy run, retuning or further sweep follows automatically.','',
            '## Figures','']
    for f in validation['figures']:lines+=['!['+f+'](figures/'+f+')','']
    (P/'RESULTS.md').write_text('\n'.join(lines).rstrip()+'\n')

def main():
    progress=json.loads((ROOT/'cpu_progress.json').read_text());assert progress['status']=='PASS' and len(progress['completed'])==1552
    for name,expected in progress['source_hashes'].items():assert sha(Path(__file__).with_name(name))==expected
    cells=[]
    for r in progress['completed']:
        path=ROOT/'cells'/(r['id']+'.json');assert sha(path)==r['sha256'];cells.append(json.loads(path.read_text()))
    cells.sort(key=lambda c:c['id']);validation=validate(cells,progress)
    for f in P.glob('CPU_*.json'):
        d=json.loads(f.read_text());validation['peak_aggregate_replay_rss_bytes']=max(validation['peak_aggregate_replay_rss_bytes'],d.get('peak_aggregate_rss_bytes',0))
    grid,comp,sub,seed,same,labels=analyze(cells);groups=aggregates(cells,comp,labels);validation['figures']=figures(cells,comp,sub,same)
    expected={'grid_points':3104,'policy_comparisons':1024,'substitution_comparisons':768,'seed_audit':16,'same_global_comparisons':576,'group_summary':8}
    for name,count in expected.items():
        rows=json.loads((P/(name+'.json')).read_text());csvrows=list(csv.DictReader((P/(name+'.csv')).open()));assert len(rows)==len(csvrows)==count
        for a,b in zip(rows,csvrows):
            for k,v in a.items():
                if isinstance(v,(list,dict)):assert json.loads(b[k])==v
                else:assert b[k]==('' if v is None else str(v))
        validation['csv_json_checks'][name]=count
    validation['label_counts']={k:len(v) for k,v in labels.items()};write('validation.json',validation);report(cells,comp,sub,seed,same,labels,groups,validation)
    print(json.dumps(validation,indent=2),flush=True)
if __name__=='__main__':main()
