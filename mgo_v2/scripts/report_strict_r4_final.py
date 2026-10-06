"""Close R8 on owner request and expose every completed R4 timing sample."""
from strict_headroom_common import *
import csv,statistics

def main():
 cells=[];aggregate=[];rank_rows=[];source_files={};selection=[];warm=[]
 for batch in (16,64):
  for context in (256,512):
   d=ROOT/'physical'/f'S2D32_R4_B{batch}_L{context}';state=json.loads((d/'status.json').read_text());assert state['status']=='PASS';x=state['result'];cells.append(x)
   source_files[str(d/'result.json')]=sha(d/'result.json')
   for policy,samples in x['samples'].items():
    for repeat,row in enumerate(samples,1):
     aggregate.append(dict(batch=batch,context=context,policy=policy,repeat=repeat,**row))
     rr=[]
     for rank in range(4):
      p=d/f'{policy}_r{repeat}_measure_rank{rank}.json';v=json.loads(p.read_text());assert v['status']==v['validation']['status']=='PASS' and v['validation']['no_compile'] and v['validation']['tokens_match_warm']
      assert v['validation']['observed']==v['validation']['expected'];rr.append(v);source_files[str(p)]=sha(p)
      rank_rows.append(dict(batch=batch,context=context,policy=policy,repeat=repeat,rank=rank,physical_gpu=[0,1,4,5][rank],TTFT=v['TTFT'],TPOT=v['TPOT'],E2E=v['E2E'],prefill_copies=v['validation']['prefill_H2D_copies'],decode_copies=v['validation']['decode_H2D_copies']))
     for metric in ['TTFT','TPOT','E2E']:assert max(v[metric] for v in rr)==row[metric]
    for rank in range(4):
     p=d/f'{policy}_warm_rank{rank}.json';warm.append(dict(stage='S2_WARM',batch=batch,context=context,policy=policy,rank=rank,source=str(p),receipt=json.loads(p.read_text())))
 for d in sorted((ROOT/'physical').glob('S1_R4_*')):
  x=json.loads((d/'result.json').read_text());assert x['status']=='PASS'
  for pair in x['results']:selection.append(pair)
  for p in d.glob('*_warm_rank*.json'):warm.append(dict(stage='S1_WARM',source=str(p),receipt=json.loads(p.read_text())))
 result=dict(status='R4_COMPLETE_R8_STOPPED_BY_OWNER',final_cells=cells,final_aggregate_samples=aggregate,final_rank_samples=rank_rows,S1_selection_pairs=selection,warm_receipts=warm,source_sha256=source_files,counts=dict(final_cells=4,primary_policy_runs=len(aggregate),primary_rank_receipts=len(rank_rows),selection_pairs=len(selection),selection_policy_runs=2*len(selection),final_warm_policy_runs=12,untimed_BR_capture_runs=4,diagnostic_runs=0,static_runs=0),scope='All completed R4 strict selected-seed results. S1 selection and warm/capture are not primary samples. R8 final validation canceled; partial R8 artifacts retained separately. No cherry-picking. No additional runs queued.')
 write(PACKET/'R4_COMPLETE_RECORDS.json',result)
 for name,rows in [('R4_PRIMARY_REPEATS.csv',aggregate),('R4_PRIMARY_RANKS.csv',rank_rows)]:
  with (PACKET/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
 lines=['# Completed R4 strict prefill + decode32 results','','Owner stopped further R8 validation. R4 final validation is complete: four conditions, three policies,33 primary runs (132 rank receipts). All primary repeats are retained. S1 had32 paired selection candidates (64 one-shot policy timings); those are stored separately and excluded from final estimates. Static remains on hold; no diagnostic passes were run.','','Units: TTFT/E2E seconds; TPOT seconds per decode step. Metrics are independently maximized over four ranks, so aggregate TTFT +32*TPOT need not equal aggregate E2E exactly.32 decode steps follow the prefill output. Same policy across both phases; cache carryover retained. GPUs0/1/4/5, C30, BF16.','','## Every primary repeat','']
 for x in cells:
  b=x['spec']['batch'];l=x['spec']['context'];lines += [f'### B{b} / input{l}','','| Policy | Repeat | TTFT s | TPOT s/token | E2E s |','|---|---:|---:|---:|---:|']
  for p,v in x['samples'].items():
   for i,z in enumerate(v,1):lines.append(f"| {p} | {i} | {z['TTFT']:.6f} | {z['TPOT']:.6f} | {z['E2E']:.6f} |")
  lines+=['']
 lines+=['## Final gains relative to BR','','Two repeats use their mean; three use their median independently per metric. No latency-based exclusions.','','| B | Input | Candidate | TTFT gain | TPOT gain | E2E gain |','|---:|---:|---|---:|---:|---:|']
 for x in cells:
  for p,g in x['gains'].items():lines.append(f"| {x['spec']['batch']} | {x['spec']['context']} | {p} | {100*g['TTFT']:+.3f}% | {100*g['TPOT']:+.3f}% | {100*g['E2E']:+.3f}% |")
 lines+=['','Third-repeat reasons: B16/L256 BR TTFT2.345% and LA TTFT2.379%; B64/L256 LA TTFT3.641%; B64/L512 LA_CA_NEAR TTFT2.528%. All policies in each affected cell received the same third repeat. B16/L512 stopped after two. None met the >5% instability criterion, but sub-percent gains can be comparable to repeat variability and are not statistical confidence claims.','','All CPU copy/state checks,33-token parity, both1584-barrier counters per rank and no compilation during primary measurements passed. H2D durations were not measured separately, so rank-specific contention cannot be determined.','','This is deliberately BR-adversarial, prefill-selected best-seed headroom, not average-case performance. TPOT includes policy-specific prefill cache-state carryover. R4 LA_CA_NEAR TTFT gains span0.90–4.02%; TPOT0.71–0.96%; E2E0.80–1.25%. LA has the smaller TPOT/E2E in B64/L512.','','Raw precision, full repeat ranges (derivable from all samples), all132 rank measurements, warm receipts, S1 selection pairs and source hashes are preserved in R4_COMPLETE_RECORDS.json. CSV files provide the33 primary and132 rank rows.','']
 (PACKET/'R4_COMPLETE_RESULTS.md').write_text('\n'.join(lines))
 print(json.dumps(result['counts']))
if __name__=='__main__':main()
