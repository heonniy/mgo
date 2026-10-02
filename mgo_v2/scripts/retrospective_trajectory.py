#!/usr/bin/env python3
"""Read only the completed study; no GPU execution."""
import csv
import json
from pathlib import Path
import statistics
import numpy as np
p = Path(__file__).resolve().parents[1] / 'experiments'
o = p / 'admission_trajectory_controller_breakdown_20261002'
rows = json.loads((p/'local_remote_e2e_impact_20261001/e2e_repeats.json').read_text())['rows']
rows = [r for r in rows if r['policy'] in ('random','hungarian_current')]
fields = ['tpot_seconds','generation_seconds','controller_seconds','physical_expert_h2d_bytes','fetches','reloads','remote_pairs','rank_token_cv','rank_token_max_mean']
summary=[]
for world,batch,policy in sorted({(r['world'],r['local_batch'],r['policy']) for r in rows}):
    group=[r for r in rows if (r['world'],r['local_batch'],r['policy'])==(world,batch,policy)]
    summary.append(dict(world=world,local_batch=batch,policy=policy,repeats=len(group),
                        **{f:statistics.median(r[f] for r in group) for f in fields}))
(o/'retrospective_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (o/'retrospective_summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
(o/'retrospective_correlations.json').write_text(json.dumps(dict(scope='Descriptive only; 50 repeated rows, 10 conditions, correlated predictors and world/batch confounding; no causality',
    row_count=len(rows),pearson_with_generation={f:float(np.corrcoef([r[f] for r in rows],[r['generation_seconds'] for r in rows])[0,1]) for f in fields}),indent=2)+'\n')
print('Retrospective: 50 repeats / 10 conditions; complete before new GPU work.')
