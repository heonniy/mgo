"""Fixed stability-triplet selection; preserve unconditional five-sample statistics."""
import itertools,json,statistics
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'experiments/main_table_global_workload_20261006/expanded_matrix'
ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')
METRICS=('TTFT','TPOT','E2E')
def select(rows):
 assert len(rows)==5 and [r['repeat'] for r in rows]==list(range(1,6))
 def score(group):
  spreads=[(max(r[k] for r in group)-min(r[k] for r in group))/statistics.mean(r[k] for r in group) for k in METRICS]
  return max(spreads),sum(spreads),tuple(r['repeat'] for r in group)
 group=min(itertools.combinations(rows,3),key=score)
 def stats(rr):return {k:dict(mean=statistics.mean(r[k] for r in rr),sample_sd=statistics.stdev(r[k] for r in rr),min=min(r[k] for r in rr),max=max(r[k] for r in rr)) for k in METRICS}
 return dict(selected_ids=[r['repeat'] for r in group],selected=stats(group),all_five=stats(rows),selected_max_relative_range=score(group)[0],stability='UNSTABLE' if score(group)[0]>.05 else 'WITHIN_5_PERCENT',all_samples=rows)
def main():
 queue=json.loads((P/'QUEUE.json').read_text());report=[]
 for job in queue['jobs']:
  out=ROOT/job['label'];status=out/'status.json'
  if not status.exists():continue
  state=json.loads(status.read_text())
  if state['status']!='PASS':continue
  rows=[json.loads((out/f'repeat{r}.json').read_text()) for r in range(1,6)]
  assert all(r['status']=='PASS' and r['output_tokens']==64 and not r['smoke'] for r in rows)
  report.append(dict(cell=job['cell'],system=job['system'],raw=str(out),source_commit=state['source_commit'],**select(rows)))
 (P/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
 lines=['# Expanded R4 results (partial until all48 finish)','', 'One common triplet minimizes maximum relative range over TTFT/TPOT/E2E. Selection is descriptive; excluded values remain valid observations. Seconds, sample standard deviation.','', '|Cell|System|Selected repeats|TTFT|TPOT|E2E|Selected spread/status|','|---|---|---|---|---|---|---|']
 for r in report:
  values=[f"{r['selected'][k]['mean']:.6f} ± {r['selected'][k]['sample_sd']:.6f}" for k in METRICS]
  lines.append('|'+ '|'.join([r['cell'],r['system'],str(r['selected_ids']),*values,f"{r['selected_max_relative_range']:.2%} {r['stability']}"])+'|')
 lines+=['','## All five samples and unconditional summaries','']
 for r in report:
  lines += [f"### {r['cell']} / {r['system']}",'', '|Repeat|TTFT|TPOT|E2E|','|---|---|---|---|']
  for row in r['all_samples']:lines.append('|'+ '|'.join([str(row['repeat'])]+[f'{row[k]:.6f}' for k in METRICS])+'|')
  lines+=['']
  for k,v in r['all_five'].items():lines.append(f"- {k}, all5 mean±SD: {v['mean']:.6f} ± {v['sample_sd']:.6f}; range [{v['min']:.6f}, {v['max']:.6f}]")
  lines+=['',f"Raw receipts: {r['raw']}",'']
 (P/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 print(f'{len(report)}/48 complete')
if __name__=='__main__':main()
