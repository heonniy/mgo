#!/usr/bin/env python3
"""Validate and publish the frozen CPU placement accounting study."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import resource
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
import csv,gzip,hashlib,json,statistics,time
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'experiments/future_rank_affinity_placement_20261003'
OLD=P.parent/'fetch_comm_pareto_p2p_20261002'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,x):(P/name).write_text(json.dumps(x,indent=2)+'\n')
def csvout(name,rows):
    with (P/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
def dominates(a,b):
    x=(a['peer_activation_bytes'],a['expert_h2d_bytes']);y=(b['peer_activation_bytes'],b['expert_h2d_bytes'])
    return x!=y and all(u<=v for u,v in zip(x,y))

def main():
    manifest=json.loads((P/'execution_manifest.json').read_text())
    for group in ('source_sha256','baseline_sha256'):
        for path,h in manifest[group].items():assert sha(path)==h,path
    points=[];runs=[];checks=[]
    for batch in (8,16,32):
        a=json.loads((P/f'B{batch}.json').read_text());b=json.loads((P/f'B{batch}_verify.json').read_text())
        assert a['status']==b['status']=='PASS' and len(a['points'])==len(b['points'])==6
        assert [r['policy'] for r in a['points']]==['F','O0','OH1','OH2','OH4','OHremaining']
        for x,y in zip(a['points'],b['points']):
            assert {k:v for k,v in x.items() if not k.endswith('_receipt')}=={k:v for k,v in y.items() if not k.endswith('_receipt')}
            for r in (x,y):
                for key in ('event_receipt','lifetime_receipt'):
                    rec=r[key];assert Path(rec['path']).stat().st_size==rec['bytes'] and sha(rec['path'])==rec['sha256']
            assert x['events']==384 and x['total_fetches']==x['first_copy_fetches']+x['reload_fetches']
            assert x['expert_h2d_bytes']==x['total_fetches']*9437184
            assert x['peer_activation_bytes']==x['dispatch_bytes']+x['combine_bytes']
            assert x['replica_fetches']==x['duplicate_slot_peak']==0
            with gzip.open(x['event_receipt']['path'],'rt') as f:event_rows=[json.loads(line) for line in f]
            assert [r['event'] for r in event_rows]==list(range(48,432))
            for key in ('first_copy_fetches','reload_fetches','total_fetches','expert_h2d_bytes','peer_activation_bytes','dispatch_bytes','combine_bytes','remote_token_rank_pairs','owner_changes','owner_attributable_peer_bytes_saved'):
                assert sum(r['row'][key] for r in event_rows)==x[key]
            assert sum(len(r['decisions']) for r in event_rows)==x['owner_decisions']
            assert all(all(n<=cap for n,cap in zip(r['row']['rank_unique'],[461,461,461,460])) for r in event_rows)
            with gzip.open(x['lifetime_receipt']['path'],'rt') as f:life=json.load(f)
            assert len(life)==x['owner_decisions']==x['total_fetches']
            assert sum(r['changed'] for r in life)==x['owner_changes']
            eligible=[r for r in life if r['next_demand_event'] is not None]
            assert len(eligible)==x['next_demand_eligible']
            assert sum(r['survived_next_demand'] for r in eligible)==x['survived_next_demand']
            assert all(not r['survived_next_demand'] or r['next_demand_event']<=r['end_event'] for r in life)
            for key in ('expert_h2d_bytes','peer_activation_bytes','reload_fetches','owner_changes'):
                assert sum(r[key] for r in x['steps'])==x[key]
            checks.append(dict(batch=batch,policy=x['policy'],hash=x['deterministic_sha256'],matched=True))
        points+=a['points'];runs += [a,b]
    scalar=[{k:v for k,v in r.items() if k not in ('steps','event_receipt','lifetime_receipt')} for r in points]
    csvout('placement_points.csv',scalar);write('placement_points.json',dict(status='PASS',points=points))
    csvout('placement_step_trajectory_B8.csv',[step for r in points if r['batch']==8 for step in r['steps']])
    ownerkeys=['batch','policy','owner_decisions','owner_changes','fraction_owner_differs_F','owner_choice_histogram','evictions_per_rank','mean_unique_per_rank','peak_unique_per_rank','next_demand_eligible','survived_next_demand','fraction_survived_next_demand','changed_next_demand_eligible','changed_survived_next_demand','admissions_without_observed_next_demand','owner_attributable_peer_bytes_saved']
    owners=[{k:r[k] for k in ownerkeys} for r in points]
    csvout('owner_choice_summary.csv',owners);write('owner_choice_summary.json',dict(rows=owners,survival='Actual copy persists until next observed global demand for same layer/expert; censored admissions excluded.',attribution='Joint fixed-cache counterfactual replaces surviving changed-decision owners with max-current-demand owner at admission; not the whole-policy F delta.'))
    historical=json.loads((OLD/'replica_pareto_screen.json').read_text())['points']
    old=[dict(label={0:'F',.25:'K',.75:'C'}.get(r['rho'],f"rho={r['rho']}"),rho=r['rho'],**{k:r['decode'][k] for k in ('expert_h2d_bytes','peer_activation_bytes')}) for r in historical]
    primary={r['policy']:r for r in points if r['batch']==8};F=primary['F'];O=primary['O0'];frontier=[]
    for r in old:frontier.append(dict(kind='historical',label=r['label'],peer_activation_bytes=r['peer_activation_bytes'],expert_h2d_bytes=r['expert_h2d_bytes'],dominates_old_rhos=[],future_incremental_gate=False,current_gate=False,modest_gate=False))
    for policy,r in primary.items():
        dominated=[q['rho'] for q in old if q['rho']>0 and dominates(r,q)]
        incremental=policy.startswith('OH') and 10*r['peer_activation_bytes']<=9*O['peer_activation_bytes'] and 20*r['expert_h2d_bytes']<=21*O['expert_h2d_bytes']
        current=policy=='O0' and 10*r['peer_activation_bytes']<=9*F['peer_activation_bytes'] and 20*r['expert_h2d_bytes']<=21*F['expert_h2d_bytes']
        modest=policy!='F' and (20*r['peer_activation_bytes']<=19*F['peer_activation_bytes'] and 20*r['expert_h2d_bytes']<=21*F['expert_h2d_bytes'] or bool(dominated))
        frontier.append(dict(kind='single_copy',label=policy,peer_activation_bytes=r['peer_activation_bytes'],expert_h2d_bytes=r['expert_h2d_bytes'],dominates_old_rhos=dominated,future_incremental_gate=incremental,current_gate=current,modest_gate=modest))
    if any(r['label'].startswith('OH') and (r['future_incremental_gate'] or r['dominates_old_rhos']) for r in frontier):decision='FUTURE_AFFINITY_HEADROOM'
    elif any(r['current_gate'] for r in frontier):decision='CURRENT_ONLY_HEADROOM'
    elif any(r['modest_gate'] for r in frontier):decision='MODEST_PLACEMENT_HEADROOM'
    else:decision='NO_PLACEMENT_HEADROOM'
    for r in frontier:r['nondominated']=not any(dominates(q,r) for q in frontier)
    csvout('frontier_comparison.csv',frontier);write('frontier_comparison.json',dict(decision=decision,rows=frontier))
    deltas=[]
    for batch in (8,16,32):
        pp={r['policy']:r for r in points if r['batch']==batch}
        for source,target in [('F','O0')]+[('O0',p) for p in ('OH1','OH2','OH4','OHremaining')]:
            a,b=pp[source],pp[target]
            deltas.append(dict(batch=batch,source=source,target=target,peer_reduction=1-b['peer_activation_bytes']/a['peer_activation_bytes'],h2d_ratio=b['expert_h2d_bytes']/a['expert_h2d_bytes']))
    write('placement_deltas.json',deltas)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,(ax,zoom)=plt.subplots(1,2,figsize=(12,5),layout='constrained',gridspec_kw={'width_ratios':[1.4,1]})
    ax.plot([r['peer_activation_bytes']/2**20 for r in old],[r['expert_h2d_bytes']/2**30 for r in old],'o-',color='#777777',label='Historical replica frontier')
    for r in old:ax.annotate(r['label'],(r['peer_activation_bytes']/2**20,r['expert_h2d_bytes']/2**30),xytext=(4,6),textcoords='offset points',fontsize=8)
    for i,(name,r) in enumerate(primary.items()):
        if name=='F':continue
        x,y=r['peer_activation_bytes']/2**20,r['expert_h2d_bytes']/2**30
        color=f'C{i-1}'
        ax.scatter(x,y,s=45,label=name,color=color)
        zoom.scatter(x,y,s=45,color=color)
        zoom.annotate(name,(x,y),xytext=(-18,8 if name!='OHremaining' else -15),textcoords='offset points',fontsize=8)
    zoom.scatter(F['peer_activation_bytes']/2**20,F['expert_h2d_bytes']/2**30,color='#777777',s=45)
    zoom.annotate('F',(F['peer_activation_bytes']/2**20,F['expert_h2d_bytes']/2**30),xytext=(5,5),textcoords='offset points')
    zoom.set(xlabel='Decode peer activation traffic (MiB)',ylabel='Decode expert H2D (GiB)',title='Single-copy region',xlim=(300,323),ylim=(146,158));zoom.grid(alpha=.2)
    ax.set(xlabel='Decode peer activation traffic (MiB)',ylabel='Decode expert H2D (GiB)',title='Historical frontier');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle(f'B8 single-copy placement: {decision}')
    fig.savefig(P/'frontier_B8.png',dpi=180);fig.savefig(P/'frontier_B8.svg');plt.close(fig)
    svg=P/'frontier_B8.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    maximum=max(r['peak_rss_mib'] for r in runs)
    lines=['# Future rank-affinity single-copy placement','',f'**{decision}** — CPU byte accounting only; no TPOT/E2E timing claim.','',
        'All 18 fixed cells completed (six policies across B8/B16/B32). Each cell was independently invoked again solely for deterministic validation; full event/cache/decision hashes matched. B8 F exactly reproduced 17,635 fetches, 166,424,739,840 H2D bytes, 334,970,880 peer bytes and 23,179 remote token-rank pairs. F cache/LRU states and traffic matrices matched the independent reference at every one of 432 events per batch and invocation.','',
        '## Actual B8 coordinates','', '| Policy | H2D GiB | Peer MiB | Changed owners | Next-demand survival |', '|:---|---:|---:|---:|---:|']
    for r in primary.values():lines.append(f"| {r['policy']} | {r['expert_h2d_bytes']/2**30:.3f} | {r['peer_activation_bytes']/2**20:.3f} | {r['fraction_owner_differs_F']:.2%} | {r['fraction_survived_next_demand']:.2%} |")
    lines+=['','## Current placement versus future affinity','', '| Batch | Comparison | Peer-byte reduction | H2D ratio |', '|---:|:---|---:|---:|']
    for r in deltas:lines.append(f"| {r['batch']} | {r['source']} → {r['target']} | {r['peer_reduction']:.2%} | {r['h2d_ratio']:.4f} |")
    best=min((r for p,r in primary.items() if p.startswith('OH')),key=lambda r:(r['peer_activation_bytes'],r['expert_h2d_bytes']))
    lines+=['',f"B8 O0 reduces peer bytes by {1-O['peer_activation_bytes']/F['peer_activation_bytes']:.4%} with H2D ratio {O['expert_h2d_bytes']/F['expert_h2d_bytes']:.6f} versus F. The predeclared modest threshold is 5%; all decisions use unrounded integer comparisons. The best OH peer reduction versus O0 is {1-best['peer_activation_bytes']/O['peer_activation_bytes']:.4%} (a negative reduction means more traffic). No threshold was adjusted after observing results."]
    lines+=['',f"The descriptive lowest-peer B8 future policy is **{best['policy']}**. F→O0 measures current placement quality; only O0→OH measures incremental future-affinity value. 'Best' here means lowest peer bytes, not a latency-optimal choice. Gate evaluation checks every OH policy, not just this descriptive winner.",'',
        '![B8 frontier](frontier_B8.png)','',
        'The full per-policy gates and dominated historical rho points are in `frontier_comparison.json`. Historical K/C use replicated prefills; the new single-copy policies share F prefill. B16/B32 are trend checks only, with no invented historical K/C references.','',
        '## Interpretation and measurement boundaries','',
        '- A globally missing expert gets exactly one copy, only on a currently requesting rank. Already-resident copies never move; no duplicate, prefetch, substitution or load-balancing objective is present.',
        '- Current-event decisions proceed by expert ID with prior choices fixed and undecided misses using the F heuristic. Future costs freeze other experts at F destinations and include current plus the allowed future steps. Actual evictions and reloads follow original physical LRU.',
        '- Owner divergence compares each actual miss to its same-event F heuristic. Survival excludes admissions without an observed next demand. Attribution is an exact joint owner counterfactual on the actual cache, separate from whole-policy differences.',
        '- Eight decode steps bound lookahead and observed survival; this offline diagnostic is not globally optimal and is not an online predictor.',
        '- A positive CPU gate does not authorize new GPU/model captures, timing or a controller. Stop for owner review.','',
        '## Validation','',f'- Fourteen targeted and regression tests passed before execution. Peak process RSS: {maximum:.2f} MiB under a hard 4-GiB address-space bound.',
        '- All 18 cell pairs have identical replay hashes. Source sizes/SHA256 and cross-rank receipts passed; historical CPU frontier files are unchanged.',
        '- Single-copy residency, no resident migration, miss-only admissions, active protection, capacities and traffic transpose checks passed throughout.',
        '- No Torch/CUDA/NCCL/model work was launched. Existing model workers on GPUs 0/1/4/5 were preserved; the owner-stopped 2/3/6/7 workers were not restarted.','',
        'See [coordinates](placement_points.json), [owner accounting](owner_choice_summary.json), [frontier](frontier_comparison.json), [validation](validation.json), and [execution protocol](EXECUTION_PROTOCOL.md).','']
    (P/'RESULTS.md').write_text('\n'.join(lines))
    workerroot=Path('/home/hwlee/mgo-results/model_inference_load_20261003')
    workers=manifest['workers_before'];worker_receipts=[]
    for w in workers:
        assert 'model_inference_load.py' in Path(f"/proc/{w['pid']}/cmdline").read_bytes().decode()
        r=json.loads((workerroot/f"gpu{w['gpu']}.json").read_text());assert r['pid']==w['pid'] and time.time()-r['unix']<120
        worker_receipts.append({k:r[k] for k in ('gpu','pid','batch','iterations','unix')})
    originals=json.loads((workerroot/'processes.json').read_text())
    assert all(not Path(f"/proc/{r['pid']}").exists() for r in originals if r['gpu'] in manifest['stopped_gpu_indices'])
    write('validation.json',dict(status='PASS',decision=decision,cells=18,invocations=36,deterministic_checks=checks,independent_F_event_checks=432*6,all_replay_event_checks=432*36,cpu_only=True,new_gpu_runs=0,torch_imported=False,threads=1,address_space_limit_bytes=4*1024**3,peak_rss_mib=maximum,tests_passed=14,source_hashes_unchanged=True,historical_frontier_hashes_unchanged=True,workers_unchanged=worker_receipts,stopped_workers_not_restarted=True,summary_source_sha256=sha(Path(__file__)),outputs={f.name:sha(f) for f in P.iterdir() if f.suffix in ('.csv','.json','.svg','.png') and f.name!='validation.json'}|{'RESULTS.md':sha(P/'RESULTS.md')}))
    print(json.dumps(dict(status='PASS',decision=decision,cells=18,peak_rss_mib=maximum),indent=2))
if __name__=='__main__':main()
