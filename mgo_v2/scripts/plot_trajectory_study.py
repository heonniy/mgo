#!/usr/bin/env python3
"""Standalone scientific figures from validated rows; no synthetic measurements."""
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize_trajectory_study import OUT,ROOT,read,events,POLICIES

COLORS={'random':'#28649A','hungarian_current':'#BB681D'}
NAMES={'random':'Random','hungarian_current':'Current'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.labelcolor':'#252525','text.color':'#252525','axes.grid':True,'grid.alpha':.16,
    'figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none'})

def export(fig,name):
    for extension in ('png','svg'):
        fig.savefig(OUT/f'{name}.{extension}',dpi=160,bbox_inches='tight')
    plt.close(fig)

def main():
    assert read(OUT/'validation.json')['status']=='PASS'
    rows=read(ROOT/'cumulative_event_deltas.json')
    metrics=[('remote_pairs',1000,'Remote pairs (thousands)'),('h2d_bytes',2**30,'Expert fetch (GiB)'),
        ('reloads',1000,'Reloads (thousands)'),('evictions',1000,'Evictions (thousands)'),
        ('coverage_seconds',1,'Coverage sync (seconds)'),('eviction_seconds',1,'Eviction work (seconds)'),
        ('controller_seconds',1,'Controller (seconds)')]
    fig,axes=plt.subplots(len(metrics),3,figsize=(15,19),sharex=True,sharey='row',layout='constrained')
    for column,batch in enumerate((4,8,16)):
        for row,(metric,scale,label) in enumerate(metrics):
            ax=axes[row,column]
            for policy in POLICIES:
                group=[r for r in rows if r['batch']==batch and r['source_policy']==policy]
                ax.plot([r['event']/48 for r in group],[r['cumulative_'+metric]/scale for r in group],
                    color=COLORS[policy],linestyle='-' if policy=='random' else '--',linewidth=1.5,label=NAMES[policy]+' raw trace')
            ax.axhline(0,color='#444444',linewidth=.8)
            ax.axvline(1,color='#777777',linewidth=.6,linestyle=':')
            if column==0:ax.set_ylabel(label)
            if row==0:ax.set_title(f'R4 / local B{batch} (global {4*batch})')
            if row==len(metrics)-1:ax.set_xlabel('Forward index (0 = prefill; 1–64 = decode)')
            ax.set_xlim(0,65);ax.set_xticks([0,16,32,48,64])
    axes[0,0].legend(loc='best',frameon=False)
    fig.suptitle('Matched raw demand: cumulative Current − Random\nCounts are deterministic; CPU time sums per-event medians over five repetitions',fontsize=15)
    export(fig,'cumulative_trajectories')
    groups=[('Coverage ranking / victim',('coverage_rank_and_victim',)),
        ('Coverage sync',('coverage_sync','coverage_sync_dispatch')),
        ('Substitution',('substitution',)),('Cache / bookkeeping',('cache_mutation_and_bookkeeping','cache_keys_on_rank')),
        ('Admission work',('admission_prep','admission_cost_build','admission_assignment','admission_result','admission_policy')),
        ('Diagnostic counters',('diagnostic_accounting',))]
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    maximum_component=0.0
    for column,batch in enumerate((4,8,16)):
        data=read(ROOT/'replay'/f'b{batch}'/'complete.json')['summaries']
        for row,source in enumerate(POLICIES):
            ax=axes[row,column]
            for offset,policy in [(-.18,'random'),(.18,'hungarian_current')]:
                raw=[r for r in data if r['source_policy']==source and r['policy']==policy]
                values=[statistics.median(sum(r['times_ns'].get(k,0) for k in keys)/1e9 for r in raw) for _,keys in groups]
                maximum_component=max(maximum_component,max(values))
                ax.barh(np.arange(len(groups))+offset,values,height=.33,color=COLORS[policy],label=NAMES[policy],
                        hatch=None if policy=='random' else '//',alpha=.9)
            ax.set_yticks(np.arange(len(groups)),[g[0] for g in groups]);ax.invert_yaxis()
            ax.set_title(f'B{batch} · {NAMES[source]} raw trace');ax.set_xlabel('Median single-process CPU seconds (five repetitions)')
            ax.set_xlim(left=0)
    for ax in axes.flat:ax.set_xlim(0,maximum_component*1.08)
    axes[0,0].legend(frameon=False)
    fig.suptitle('Controller component work under identical raw routing demand\nExclusive spans; instrumented diagnostics, not uninstrumented speedup estimates',fontsize=15)
    export(fig,'controller_components')
    survival=read(OUT/'next_use_survival.json')
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained',sharey=True)
    for ax,batch in zip(axes,(4,8,16)):
        for offset,policy in [(-.18,'random'),(.18,'hungarian_current')]:
            group=[next(r for r in survival if r['mode']=='replay' and r['batch']==batch and r['source_policy']==source and r['policy']==policy) for source in POLICIES]
            values=[r['next_use_survival']*100 for r in group]
            bars=ax.bar(np.arange(2)+offset,values,width=.33,color=COLORS[policy],label=NAMES[policy],hatch=None if policy=='random' else '//')
            ax.bar_label(bars,fmt='%.1f%%',padding=3,fontsize=9)
        ax.set_xticks([0,1],['Random trace','Current trace']);ax.set_title(f'R4 / B{batch}');ax.set_ylim(bottom=0)
    axes[0].set_ylabel('Resident at next raw demand (%)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Admission next-use survival under matched demand\nDenominator: admissions with a later raw demand; end-of-trace censored admissions excluded',fontsize=14)
    export(fig,'next_use_survival')
    fig,axes=plt.subplots(2,3,figsize=(15,9),sharex=True,sharey='row',layout='constrained')
    for column,batch in enumerate((4,8,16)):
        for source in POLICIES:
            paired={p:events(ROOT/'replay'/f'b{batch}'/f'{source}-to-{p}-rep0-events.jsonl') for p in POLICIES}
            for row,field in enumerate(('rank_tokens','rank_expert_rows')):
                ax=axes[row,column]
                xs=[];ys=[]
                for step in range(1,65):
                    groups={p:paired[p][step*48:(step+1)*48] for p in POLICIES}
                    remote={p:sum(r['remote_pairs'] for r in group) for p,group in groups.items()}
                    load={p:sum(max(r[field]) for r in group) for p,group in groups.items()}
                    assert remote['random']>0 and load['random']>0
                    xs.append(100*(remote['hungarian_current']/remote['random']-1))
                    ys.append(100*(load['hungarian_current']/load['random']-1))
                ax.scatter(xs,ys,s=23,alpha=.7,color=COLORS[source],marker='o' if source=='random' else '^',label=NAMES[source]+' raw trace')
                ax.axhline(0,color='#444444',linewidth=.8);ax.axvline(0,color='#444444',linewidth=.8)
                ax.set_title(f'R4 / B{batch}');ax.set_xlabel('Remote-pair change (%)')
    axes[0,0].set_ylabel('Change in sum of per-layer maxima (%)\nDeduplicated dispatched tokens')
    axes[1,0].set_ylabel('Change in sum of per-layer maxima (%)\nPlanned expert GEMM rows')
    axes[0,0].legend(frameon=False)
    fig.suptitle('Communication versus rank work under matched demand\nCurrent relative to Random; each point is one decode step (48 layer events), not a GPU timing',fontsize=14)
    export(fig,'communication_load')
    (OUT/'figure_data_validation.json').write_text(json.dumps(dict(status='PASS',data_rows=len(rows),
        figures=['cumulative_trajectories','controller_components','next_use_survival','communication_load'],
        caveat='Rendered visual inspection is recorded separately; no E2E speedup claims from diagnostic figures.'),indent=2)+'\n')

if __name__=='__main__':main()
