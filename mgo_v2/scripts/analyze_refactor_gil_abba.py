"""Bounded diagnostic intervention; never a primary policy gain estimator."""
import argparse,json
from pathlib import Path
from analyze_refactor_host_variation import summarize

def analyze(root):
 state=json.loads((root/'status.json').read_text())
 assert state['status']=='PASS' and (state['case'].get('gil_switch_abba') or state['case'].get('thread_affinity_abba'))
 samples=[];ranks={}
 for rep in range(1,5):
  rows=[json.loads((root/f'diagnostic_r{rep}_rank{r}.json').read_text()) for r in range(4)]
  assert all(x['status']=='PASS' and not x['primary_timing'] and x['repeat']==rep and x['rank']==r for r,x in enumerate(rows))
  intervals={x['switch_interval_s'] for x in rows};assert len(intervals)==1
  samples.append(dict(repeat=rep,switch_interval_s=intervals.pop(),thread_placement=[x.get('thread_placement') for x in rows],**{m:max(x['metrics'][m] for x in rows) for m in ('E2E_wall','TPOT')}))
  for r,x in enumerate(rows):ranks.setdefault(r,[]).append(dict(repeat=rep,host=summarize(x),usage=x['process_usage'],controller=x['controller_counters'],scheduler=x['scheduler_metrics']))
 if state['case'].get('gil_switch_abba'):
  assert samples[0]['switch_interval_s']==samples[3]['switch_interval_s'] and samples[1]['switch_interval_s']==samples[2]['switch_interval_s']==.001
 else:
  for i,s in enumerate(samples):assert all(x['isolated']==(i in (1,2)) for x in s['thread_placement'])
 for rows in ranks.values():assert all(x['controller']==rows[0]['controller'] for x in rows)
 conditions={}
 for label,indices in [('original',[0,3]),('intervention',[1,2])]:
  conditions[label]={}
  for metric in ('E2E_wall','TPOT'):
   values=[samples[i][metric] for i in indices];mean=sum(values)/2
   conditions[label][metric]=dict(samples=values,mean=mean,relative_difference=abs(values[0]-values[1])/mean,range=[min(values),max(values)])
 return dict(primary_timing=False,samples=samples,conditions=conditions,ranks=ranks,interpretation='ABBA two diagnostic generations per setting. All samples retained. Within-setting stability and between-setting association only; instrumentation and temporal interference remain possible. No primary policy gain claim.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('capture',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=analyze(a.capture);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['conditions'],indent=2))
