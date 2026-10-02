#!/usr/bin/env python3
"""Explain prespecified matched-demand fetch-divergence events using cache states."""
import bisect
import json
from pathlib import Path
from summarize_trajectory_study import ROOT,OUT,POLICIES,events,read


def future_use(rows,event,key):
    exact=anchor=0;next_raw=None;evicted=None
    for row in rows[event+1:]:
        if row['layer']==key[0]:
            if next_raw is None and key[1] in row['raw_sources']:
                next_raw=row['event']-event
            if key[1] in dict(row['owners']):
                exact+=key[1] in row['raw_sources'] and key[1] not in dict(row['substitutes'])
                anchor+=sum(target==key[1] for source,target in row['substitutes'])
        if any((a[0],a[1])==key for a in row['evictions']):
            evicted=row['event']-event
            break
    return dict(next_raw_demand_distance_before_eviction=next_raw,future_exact_events_before_eviction=exact,
                future_substitute_sources_before_eviction=anchor,eviction_distance=evicted,
                followup_right_censored=evicted is None)


def main():
    examples=[]
    for batch in (4,8,16):
        root=ROOT/'replay'/f'b{batch}'
        for source in POLICIES:
            rows={p:events(root/f'{source}-to-{p}-rep0-events.jsonl') for p in POLICIES}
            snapshots={p:{r['event']:r for r in read(root/f'{source}-to-{p}-snapshots.json')} for p in POLICIES}
            for index in read(root/f'{source}-selected-events.json'):
                delta=len(rows['hungarian_current'][index]['admissions'])-len(rows['random'][index]['admissions'])
                states={p:{(r['layer'],r['expert']):r for r in snapshots[p][index]['residents']} for p in POLICIES}
                preferred='hungarian_current' if delta<0 else 'random'
                other='random' if preferred=='hungarian_current' else 'hungarian_current'
                candidates=sorted(set(states[preferred])-set(states[other]))
                if not candidates:
                    candidates=sorted(states[preferred])
                # Prefer a resident with an observed next raw use before eviction;
                # selection affects only explanatory examples, never timing/policy.
                focus=candidates[0]
                for key in candidates[:128]:
                    if future_use(rows[preferred],index,key)['next_raw_demand_distance_before_eviction'] is not None:
                        focus=key;break
                examples.append(dict(batch=batch,source_policy=source,event=index,step=index//48,layer=index%48,
                    direction='fewer Current fetches' if delta<0 else 'more Current fetches' if delta>0 else 'equal fetches',
                    delta_fetches=delta,delta_remote_pairs=rows['hungarian_current'][index]['remote_pairs']-rows['random'][index]['remote_pairs'],
                    focus=list(focus),states={p:dict(resident=states[p].get(focus),
                        future=future_use(rows[p],index,focus) if focus in states[p] else None,
                        exact_execution_owner=dict(rows[p][index]['owners']).get(focus[1]) if index%48==focus[0] else None,
                        cache_sha256=snapshots[p][index]['cache_sha256']) for p in POLICIES},
                    raw_snapshots={p:str(root/f'{source}-to-{p}-snapshots.json') for p in POLICIES}))
    (OUT/'state_examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    lines=['# Matched-demand cache-state examples','',
        'Selection: for each batch/source stream, the five largest positive and five largest negative instantaneous admission-count deltas after prefill; earliest event wins ties. Snapshots are captured on replay repetition 1 at the same event indexes for both policies. They retain every resident key/rank, execution owners, gate score and independently computed Coverage damage. Selection is descriptive, not policy tuning.','',
        '“Positive/negative” below refers to fewer/more Current fetches at that event, not an E2E performance win/loss. These state examples do not establish the total future value of an individual placement. Next-demand distance is in global layer events; absence before eviction is not proof of no later demand.','',
        '| B | Raw source | Event (step/layer) | Current − Random fetches | Focus (layer, expert) | Random state / future uses | Current state / future uses |',
        '|---:|---|---|---:|---|---|---|']
    for e in examples:
        display=[]
        for policy in POLICIES:
            state=e['states'][policy];r=state['resident'];future=state['future']
            display.append('absent' if r is None else f"rank {r['rank']}, gate {r['gate_score']:.5f}, damage {r['coverage_damage']}; next {future['next_raw_demand_distance_before_eviction']}, exact {future['future_exact_events_before_eviction']}, anchor {future['future_substitute_sources_before_eviction']}, evict {future['eviction_distance']}")
        lines.append(f"| {e['batch']} | {e['source_policy']} | {e['event']} ({e['step']}/{e['layer']}) | {e['delta_fetches']:+d} | {tuple(e['focus'])} | {display[0]} | {display[1]} |")
    pos=sum(e['delta_fetches']<0 for e in examples);neg=sum(e['delta_fetches']>0 for e in examples)
    assert pos>=5 and neg>=5
    lines+=['',f'Retained {pos} fewer-fetch and {neg} more-fetch examples. Full state and raw snapshot paths: [state_examples.json](state_examples.json).']
    (OUT/'state_examples.md').write_text('\n'.join(lines)+'\n')
    print(f'State examples: {pos} positive / {neg} negative')

if __name__=='__main__':main()
