#!/usr/bin/env python3
"""Compact, source-backed byte frontiers and mechanism accounting; no timing claim."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import resource
resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
import argparse,csv,hashlib,json,math
from collections import Counter
from fractions import Fraction
from pathlib import Path
from price_envelope import serial_envelope
P=Path(__file__).resolve().parents[1]/'experiments/cache_eviction_substitution_20261003'
ROOT=Path('/home/hwlee/mgo-results/cache_eviction_substitution_20261003')
ANCHOR=Fraction(7435008,17693)
H='expert_h2d_bytes';C='peer_activation_bytes'

def write(name,obj):(P/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def table(name,rows):
 write(name+'.json',rows)
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with (P/(name+'.csv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  w.writerows({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)
def meta(c):return {k:c[k] for k in ('id','batch','cache_ratio','eviction','substitution','rho','fixed_duplicate_cap','duplicate_cap')}
def flat(c):
 row=meta(c)
 for scope in ('full','decode'):
  row.update({scope+'_'+k:v for k,v in c[scope].items()})
 return row
def dominates(a,b):return a[H]<=b[H] and a[C]<=b[C] and (a[H]<b[H] or a[C]<b[C])
def ratio(a,b):return a/b if b else (1. if a==0 else None)
def key(c):return c['batch'],c['cache_ratio'],c['eviction'],c['substitution'],c['rho'],c['fixed_duplicate_cap']

def analyze(cells):
 grid=[c for c in cells if c['fixed_duplicate_cap'] is None];fixed=[c for c in cells if c['fixed_duplicate_cap'] is not None]
 lookup={key(c):c for c in cells};assert len(lookup)==len(cells)
 labels={name:[] for name in ('CACHE_RELIEF','EVICTION_HEADROOM','SUBSTITUTION_SYSTEM_HEADROOM','REPLICATION_REGIME_SHIFT')}
 substitution=[];eviction=[];life=[];frontiers=[]
 for c in cells:
  m=meta(c);b,cache,ev,sub,rho,cap=key(c)
  for scope in ('full','decode'):
   a=c[scope]
   life.append(dict(**m,scope=scope,**{k:v for k,v in a.items() if k.startswith('replica_') or k in ('mean_duplicate_slots','peak_duplicate_slots','mean_unique_resident_experts','rank_evictions')}))
   if sub:
    off=lookup[b,cache,ev,False,rho,cap][scope]
    rec=dict(**m,scope=scope,h2d_avoided_bytes=off[H]-a[H],h2d_ratio=ratio(a[H],off[H]),peer_ratio=ratio(a[C],off[C]),**{k:v for k,v in a.items() if k.startswith(('substituted_','accepted_similarity','tier_')) or k in ('protected_exact_misses','residual_exact_misses')})
    substitution.append(rec)
    if scope=='decode' and cap is None and 10*a[H]<=9*off[H] and 20*a[C]<=21*off[C]:labels['SUBSTITUTION_SYSTEM_HEADROOM'].append(c['id'])
   if ev!='lru':
    base=lookup[b,cache,'lru',sub,rho,cap][scope]
    qualifies=dominates(a,base) or (10*a[H]<=9*base[H] and 20*a[C]<=21*base[C])
    eviction.append(dict(**m,scope=scope,h2d_ratio=ratio(a[H],base[H]),peer_ratio=ratio(a[C],base[C]),dominates_lru=dominates(a,base),headroom=qualifies,reload_fetches=a['reload_fetches'],rank_evictions=a['rank_evictions']))
    if scope=='decode' and cap is None and qualifies:labels['EVICTION_HEADROOM'].append(c['id'])
  if cap is not None and cache>.3:
   a=c['decode'];base=lookup[b,.3,ev,sub,rho,cap]['decode'];x=a['replica_survival48_fraction'];y=base['replica_survival48_fraction']
   survival=x is not None and y is not None and x>0 and x>=2*y
   if (20*a[H]<=17*base[H] and 10*a[C]<=11*base[C]) or survival:labels['CACHE_RELIEF'].append(dict(id=c['id'],byte_clause=20*a[H]<=17*base[H] and 10*a[C]<=11*base[C],survival_clause=survival,h2d_ratio=ratio(a[H],base[H]),peer_ratio=ratio(a[C],base[C]),survival48=x,baseline_survival48=y))
  if cap is None and b==8 and rho>0:
   base=lookup[8,.3,'lru',False,rho,None]['decode']
   if dominates(c['decode'],base):labels['REPLICATION_REGIME_SHIFT'].append(dict(id=c['id'],reason='dominates matched-rho historical B8'))
 groups=sorted({(c['batch'],c['cache_ratio'],c['eviction'],c['substitution']) for c in grid})
 for b,cache,ev,sub in groups:
  rows=[c for c in grid if key(c)[:4]==(b,cache,ev,sub)]
  assert len(rows)==5
  for scope in ('decode','full'):
   env=serial_envelope([(str(c['rho']),c[scope][H],c[scope][C]) for c in rows])
   first=next((r for r in env['regions'] if any(float(w)>0 for w in r['winners'])),None)
   price=first['start'] if first else None
   # All grid prices are integer/fraction based. Numeric display never drives labels.
   qualified=price is not None and Fraction(price['numerator'],price['denominator'])<=ANCHOR/4
   nondominated=[c['rho'] for c in rows if not any(dominates(other[scope],c[scope]) for other in rows)]
   row=dict(batch=b,cache_ratio=cache,eviction=ev,substitution=sub,scope=scope,lambda_first=price['decimal'] if price else None,lambda_first_exact=price,first_nonzero_winners=first['winners'] if first else [],rho_nondominated=nondominated,envelope=env)
   frontiers.append(row)
   if scope=='decode' and qualified:labels['REPLICATION_REGIME_SHIFT'].append(dict(batch=b,cache_ratio=cache,eviction=ev,substitution=sub,reason='lambda_first <= historical anchor / 4',lambda_first=price['decimal'],cross_batch_anchor=b==32))
 if not any(labels.values()):labels={'NO_RESCUE':True}
 # Fixed-460/cache30 must equal the mathematically identical rho=.25 cells.
 equal=[]
 for c in fixed:
  if c['cache_ratio']==.3:
   other=lookup[8,.3,c['eviction'],c['substitution'],.25,None]
   assert all(c[k]==other[k] for k in ('full','decode','final_state_sha256'))
   equal.append(c['id'])
 historical=next(r for r in frontiers if (r['batch'],r['cache_ratio'],r['eviction'],r['substitution'],r['scope'])==(8,.3,'lru',False,'decode'))
 assert historical['lambda_first_exact']['numerator']==ANCHOR.numerator and historical['lambda_first_exact']['denominator']==ANCHOR.denominator
 table('grid_points',[flat(c) for c in grid]);table('fixed_duplicate_control',[flat(c) for c in fixed]);table('substitution_summary',substitution);table('eviction_summary',eviction);table('replica_lifecycle',life);table('frontier_summary',frontiers)
 write('interpretation.json',labels)
 return grid,fixed,frontiers,labels,equal

def figures(grid,fixed,frontiers):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 import numpy as np
 out=P/'figures';out.mkdir(exist_ok=True)
 colors={'lru':'#3875b5','gate':'#dd7f2c','coverage':'#29976c'}
 styles={0:'-',.125:':',.25:'--',.5:'-.',.75:':'}
 saved=[]
 def save(fig,name):
  fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/(name+'.png'),dpi=150);plt.close(fig);saved.append(str(out/(name+'.png')))
 def panel_title(b,s):return f'B{b}, substitution {"ON" if s else "OFF"}'
 def series(rows,ax,metric,scale=1,rhos=(0,.25,.75)):
  for ev,color in colors.items():
   for rho in rhos:
    rr=sorted([r for r in rows if r['eviction']==ev and r['rho']==rho],key=lambda r:r['cache_ratio'])
    y=[r['decode'][metric] for r in rr]
    ax.plot([r['cache_ratio']*100 for r in rr],[v/scale if v is not None else np.nan for v in y],color=color,linestyle=styles[rho],marker='.',label=f'{ev} rho={rho}')
  ax.set_xlabel('Cache (%)');ax.grid(alpha=.2)
 fig,axes=plt.subplots(2,2,figsize=(10,7))
 for i,b in enumerate((8,32)):
  for j,sub in enumerate((False,True)):
   ax=axes[i,j]
   for ev,color in colors.items():
    rr=sorted([r for r in frontiers if (r['batch'],r['substitution'],r['eviction'],r['scope'])==(b,sub,ev,'decode')],key=lambda r:r['cache_ratio'])
    ax.plot([r['cache_ratio']*100 for r in rr],[r['lambda_first'] if r['lambda_first'] is not None else np.nan for r in rr],color=color,marker='o',label=ev)
   ax.axhline(float(ANCHOR),color='gray',linestyle=':',label='historical');ax.axhline(float(ANCHOR/4),color='gray',linestyle='--',label='quarter anchor')
   ax.set(title=panel_title(b,sub),xlabel='Cache (%)',ylabel='First nonzero-rho byte price');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8);fig.suptitle('Decode resource crossover; zero = replication wins immediately')
 save(fig,'lambda_first')
 fig,axes=plt.subplots(2,4,figsize=(15,7))
 for i,b in enumerate((8,32)):
  for j,cache in enumerate((.3,.4,.5,.6)):
   ax=axes[i,j]
   for ev,color in colors.items():
    for sub in (False,True):
     rr=sorted([r for r in grid if (r['batch'],r['cache_ratio'],r['eviction'],r['substitution'])==(b,cache,ev,sub)],key=lambda r:r['rho'])
     ax.plot([r['decode'][C]/2**20 for r in rr],[r['decode'][H]/2**30 for r in rr],color=color,linestyle='--' if sub else '-',marker='.',label=f'{ev} {"ON" if sub else "OFF"}')
   ax.set(title=f'B{b} cache{round(cache*100)}',xlabel='Peer MiB',ylabel='H2D GiB');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=7);fig.suptitle('Decode raw rho trajectories (all five points, including dominated points)');save(fig,'byte_frontiers')
 for metric,label,name in (('reload_fetches','Exact reload fetches','reloads'),('replica_survival48_fraction','Observed survival >=48 (lower bound)','replica_survival')):
  fig,axes=plt.subplots(2,2,figsize=(11,8))
  for i,b in enumerate((8,32)):
   for j,sub in enumerate((False,True)):
    ax=axes[i,j];series([r for r in grid if (r['batch'],r['substitution'])==(b,sub)],ax,metric,rhos=(.125,.25,.75) if name=='replica_survival' else (0,.25,.75));ax.set(title=panel_title(b,sub),ylabel=label)
  axes[0,0].legend(fontsize=6,ncol=3);fig.suptitle('Decode; selected rho curves (complete grid in CSV)');save(fig,name)
 fig,axes=plt.subplots(2,2,figsize=(10,8))
 markers={.3:'o',.4:'s',.5:'^',.6:'D'}
 for i,b in enumerate((8,32)):
  for j,sub in enumerate((False,True)):
   ax=axes[i,j]
   for ev,color in colors.items():
    for cache,marker in markers.items():
     rr=sorted([r for r in grid if (r['batch'],r['substitution'],r['eviction'],r['cache_ratio'])==(b,sub,ev,cache)],key=lambda r:r['rho'])
     ax.plot([r['decode']['mean_duplicate_slots'] for r in rr],[r['decode']['mean_unique_resident_experts'] for r in rr],color=color,marker=marker,markersize=3,label=f'{ev} {round(cache*100)}%')
   ax.set(title=panel_title(b,sub),xlabel='Mean duplicate slots',ylabel='Mean unique resident experts');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=6,ncol=3);fig.suptitle('Decode global unique coverage versus duplicate occupancy');save(fig,'unique_coverage')
 fig,axes=plt.subplots(2,2,figsize=(11,8))
 for i,b in enumerate((8,32)):
  for j,(metric,label) in enumerate((('substituted_route_fraction','Substituted raw route fraction'),('residual_exact_misses','Residual exact expert-event misses'))):
   ax=axes[i,j];series([r for r in grid if r['batch']==b and r['substitution']],ax,metric);ax.set(title=f'B{b} substitution ON',ylabel=label)
 axes[0,0].legend(fontsize=6,ncol=3);fig.suptitle('Substitution system accounting; no model quality conclusion');save(fig,'substitution')
 fig,axes=plt.subplots(2,2,figsize=(10,8))
 for ax,(metric,label,scale) in zip(axes.ravel(),((H,'H2D GiB',2**30),(C,'Peer MiB',2**20),('replica_survival48_fraction','Survival >=48 lower bound',1),('reload_fetches','Exact reloads',1))):
  for ev,color in colors.items():
   for sub in (False,True):
    rr=sorted([r for r in fixed if (r['eviction'],r['substitution'])==(ev,sub)],key=lambda r:r['cache_ratio'])
    ax.plot([r['cache_ratio']*100 for r in rr],[r['decode'][metric]/scale if r['decode'][metric] is not None else np.nan for r in rr],color=color,linestyle='--' if sub else '-',marker='.',label=f'{ev} {"ON" if sub else "OFF"}')
  ax.set(xlabel='Cache (%)',ylabel=label);ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=7);fig.suptitle('B8 fixed 460-duplicate cap: cache-relief control, decode');save(fig,'fixed460')
 return saved

def report(grid,fixed,frontiers,labels,validation):
 lookup={key(c):c for c in grid+fixed}
 rows=['# Cache, eviction and substitution: bounded system accounting','',
       'Status: complete. Labels: **'+', '.join(k for k,v in labels.items() if v)+'**.','',
       'All 264 CPU cells completed: 120 B8 fractional-rho, 24 B8 fixed-460, and 120 B32 fractional-rho. '
       'The five historical cache30/LRU/OFF points match full/decode counters and final cache hashes exactly; '
       'their S1 runs are reused, not repeated. Both diagnostic captures match historical tokens, routes, '
       'selected weights, owners and plan/cache hashes on every rank.','',
       'This is frozen-route byte accounting. Substitution changes model computation, and its autoregressive '
       'routes and accuracy were not measured. Lambda is a byte-resource price, not a timing or bandwidth multiplier.','',
       '## Decode first replication crossover','',
       '| Batch | Eviction | Substitution | Cache30 | Cache40 | Cache50 | Cache60 |',
       '|---|---|---|---:|---:|---:|---:|']
 for b in (8,32):
  for ev in ('lru','gate','coverage'):
   for sub in (False,True):
    values=[]
    for cache in (.3,.4,.5,.6):
     r=next(r for r in frontiers if (r['batch'],r['cache_ratio'],r['eviction'],r['substitution'],r['scope'])==(b,cache,ev,sub,'decode'))
     values.append('none' if r['lambda_first'] is None else f"{r['lambda_first']:.3f}")
    rows.append(f'| B{b} | {ev} | {"ON" if sub else "OFF"} | '+ ' | '.join(values)+' |')
 rows+=['','Zero means a nonzero-rho policy wins for arbitrarily small nonnegative price. '
         'The full rational envelopes, including dominated rho points, are in frontier_summary.json.','',
         '## Representative matched K-budget and fixed-cap coordinates','',
         '| Batch | Cache | Eviction | Sub | Budget | H2D GiB | Peer MiB | Reloads | Unique residents | Survival >=48 |',
         '|---|---:|---|---|---|---:|---:|---:|---:|---:|']
 for b,cache,ev,sub,rho,cap in [(b,cache,ev,sub,.25,None) for b in (8,32) for cache in (.3,.6) for ev,sub in (('lru',False),('coverage',False),('coverage',True))]+[(8,cache,'lru',False,0,460) for cache in (.4,.5,.6)]:
  c=lookup[b,cache,ev,sub,rho if cap is None else None,cap];a=c['decode'];sur=a['replica_survival48_fraction'];s='n/a' if sur is None else f'{sur:.2%}'
  rows.append(f'| B{b} | {cache:.0%} | {ev} | {"ON" if sub else "OFF"} | {"460 slots" if cap else "rho .25"} | {a[H]/2**30:.3f} | {a[C]/2**20:.3f} | {a["reload_fetches"]:,} | {a["mean_unique_resident_experts"]:.1f} | {s} |')
 rows+=['','## Predeclared labels','']
 for name,items in labels.items():
  rows.append(f'- **{name}**: '+(f'{len(items)} qualifying comparisons; full witnesses in interpretation.json.' if isinstance(items,list) else str(items)))
 rows+=['','The cache-relief label uses only matched B8 fixed-460 controls. '
         'The eviction and substitution labels use matched fractional-rho grid cells. '
         'The replication-dominance clause uses historical B8 at the same rho; '
         'B32 lambda comparisons against the B8 numeric anchor are explicitly marked. '
         'These are existence labels, not claims that every rho or workload improves.','',
         '## Lifetimes and substitution limits','',
         'Replica lifetimes are physical-copy lifetimes in global layer events. Admission-event service '
         'does not count as reuse. Lifetime percentiles include observed lower bounds for end-of-trace '
         'censored copies. Survival>=48 is the observed-survivor/all-admission lower bound; unknown '
         'short censored copies and a known-outcome fraction are included in replica_lifecycle.csv. '
         'Decode lifetime cohorts include decode-born copies, while decode service/eviction counters '
         'include all copies used or evicted in decode.','',
         'The trace contains one prefill and eight decode forwards; these are short-horizon, cold-start '
         'accounting results, not steady-state estimates. W128 is a token window: B8/global32 spans '
         'four decode events per layer, while B32/global128 spans one. Batch comparisons therefore '
         'include both changed demand and this prescribed history horizon.','',
         'Substitution counts accepted source expert-events; route/gate-mass fractions use raw selected '
         'routes and weights before target merging. Similarities are source-event weighted. '
         'H2D avoided is the signed difference against the matched OFF replay, not a timing saving.','',
         '## Validation and resource use','',
         f"- CPU peak RSS: {validation['peak_cpu_rss_mib']:.2f} MiB; hard 8-GiB address-space limit, one process and one BLAS/OMP thread.",
         f"- Capture peak process-tree RSS: {validation['capture_peak_group_rss_gib']:.2f} GiB; no OOM or memory-guard failure.",
         '- GPU 0/1/4/5 resident-model workers were restored after capture; GPU 2/3/6/7 were untouched.',
         '- All cells enforce physical capacities, duplicate caps, active protection, legal destinations and independent send/receive transpose counting.',
         '- All six cache30 fixed-460 controls equal their rho=.25 counterparts in full/decode accounting and final cache state.',
         '- Small policy tests independently check historical actions, ragged traffic, protected substitution, and every GATE/COVERAGE victim against direct global-unique coverage.',
         '- No quality evaluation, B64/B128 capture, R8, NCCL timing, or physical F/K/C was run. Stop for owner review.','',
         '## Files','',
         'Complete full/decode counters: grid_points and fixed_duplicate_control CSV/JSON. '
         'Matched mechanism comparisons: substitution_summary, eviction_summary, replica_lifecycle. '
         'Exact envelopes: frontier_summary. Interpretation witnesses: interpretation.json. '
         'Raw capture/cell receipt paths and hashes: S0_capture.json, trace_readiness.json and cell_receipts.json.','',
         '## Figures','']
 for name in ('lambda_first','byte_frontiers','reloads','replica_survival','unique_coverage','substitution','fixed460'):rows+=['![ '+name+' ](figures/'+name+'.png)','']
 (P/'RESULTS.md').write_text('\n'.join(rows)+'\n')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');a=parser.parse_args()
 if a.check_only:return
 progress=json.loads((P/'grid_progress.json').read_text());assert progress['status']=='PASS' and len(progress['completed'])==264
 for source,expected in progress['source_hashes'].items():assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==expected
 cells=[json.loads((ROOT/'cells'/(r['id']+'.json')).read_text()) for r in progress['completed']]
 assert len({c['id'] for c in cells})==264
 grid,fixed,frontiers,labels,equal=analyze(cells)
 saved=figures(grid,fixed,frontiers)
 s0=json.loads((P/'S0_capture.json').read_text());assert s0['status']=='PASS'
 statuses=[json.loads(p.read_text()) for p in (ROOT/'captures').glob('*/status.json')];assert len(statuses)==2 and all(r['status']=='PASS' for r in statuses)
 validation=dict(status='PASS',completed_cells=len(cells),fractional_B8=sum(c['batch']==8 for c in grid),fixed_B8=len(fixed),fractional_B32=sum(c['batch']==32 for c in grid),historical_five_rho_parity=json.loads((P/'S1_parity.json').read_text()),fixed460_cache30_equality=equal,peak_cpu_rss_mib=max(progress['peak_rss_mib'],max(c['peak_rss_mib'] for c in cells)),capture_peak_group_rss_gib=max(m.get('group_rss_bytes',0) for r in statuses for m in r['memory'])/2**30,model_captures=2,capture_token_route_cache_parity=True,cpu_address_space_limit_bytes=8*2**30,cpu_processes=1,blas_omp_threads=1,cuda_visible_devices='',labels=[k for k,v in labels.items() if v],no_quality_claim=True,no_timing_claim=True,restored_workers=s0['restored_workers'],figures=saved,grid_seconds=sum(c['seconds'] for c in cells))
 write('validation.json',validation)
 write('cell_receipts.json',[dict(id=c['id'],path=str(ROOT/'cells'/(c['id']+'.json')),sha256=hashlib.sha256((ROOT/'cells'/(c['id']+'.json')).read_bytes()).hexdigest()) for c in cells])
 report(grid,fixed,frontiers,labels,validation)
 print(json.dumps({k:validation[k] for k in ('status','completed_cells','labels','peak_cpu_rss_mib','grid_seconds')}))
if __name__=='__main__':main()
