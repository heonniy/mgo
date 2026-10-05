"""Compare two diagnostic generations without treating attribution as causality."""
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path

def summarize(row):
 phases=row['host']['phases'];generation=[r for r in phases if r['phase']=='generation']
 assert len(generation)==1
 main_tid=generation[0]['tid'];groups=defaultdict(lambda:dict(wall_s=0.,cpu_s=0.,calls=0))
 for phase in phases:
  role='main' if phase['tid']==main_tid else 'staging'
  key=(role,phase['phase'])
  groups[key]['wall_s']+=phase['exclusive_wall_ns']/1e9
  groups[key]['cpu_s']+=phase['exclusive_cpu_ns']/1e9
  groups[key]['calls']+=phase['calls']
 main_total=sum(v['wall_s'] for (role,name),v in groups.items() if role=='main')
 assert abs(main_total-generation[0]['inclusive_wall_ns']/1e9)<1e-6
 return dict(generation_wall_s=generation[0]['inclusive_wall_ns']/1e9,
             generation_cpu_s=generation[0]['inclusive_cpu_ns']/1e9,
             phases={role+'/'+name:dict(**values,non_cpu_wall_s=values['wall_s']-values['cpu_s']) for (role,name),values in groups.items()},
             gc_wall_s=sum(r['wall_ns'] for r in row['host']['gc'])/1e9,
             gc_events=len(row['host']['gc']))

def analyze(root):
 state=json.loads((root/'status.json').read_text());assert state['status']=='PASS' and state['primary_timing'] is False
 by_rank={};sources={};primary_like=[]
 for repeat in (1,2):
  primary_like.append({})
  for rank in range(4):
   f=root/f'diagnostic_r{repeat}_rank{rank}.json';raw=f.read_bytes();row=json.loads(raw)
   assert row['status']=='PASS' and row['primary_timing'] is False and row['rank']==rank and row['repeat']==repeat
   sources[str(f)]=hashlib.sha256(raw).hexdigest();by_rank.setdefault(rank,[]).append((row,summarize(row)))
   primary_like[-1][rank]={m:row['metrics'][m] for m in ('E2E_wall','TPOT')}
 ranks={}
 for rank,pair in by_rank.items():
  first,second=[x[1] for x in pair];assert pair[0][0]['case']==pair[1][0]['case']
  delta={}
  for key in set(first['phases'])|set(second['phases']):
   a=first['phases'].get(key,dict(wall_s=0,cpu_s=0,non_cpu_wall_s=0,calls=0));b=second['phases'].get(key,dict(wall_s=0,cpu_s=0,non_cpu_wall_s=0,calls=0))
   delta[key]={field:b[field]-a[field] for field in a}
  ranks[rank]=dict(repeats=[first,second],delta_second_minus_first=dict(generation_wall_s=second['generation_wall_s']-first['generation_wall_s'],generation_cpu_s=second['generation_cpu_s']-first['generation_cpu_s'],gc_wall_s=second['gc_wall_s']-first['gc_wall_s'],phases=delta),logical_work_identical=pair[0][0]['controller_counters']==pair[1][0]['controller_counters'],scheduler_identical=pair[0][0]['scheduler_metrics']==pair[1][0]['scheduler_metrics'])
 return dict(status='PASS',primary_timing=False,case=state['case'],ranks=ranks,diagnostic_metrics=[{metric:max(v[metric] for v in rep.values()) for metric in ('E2E_wall','TPOT')} for rep in primary_like],source_sha256=sources,interpretation='Two same-policy instrumented generations only. Main exclusive phase totals reconcile to generation wall time; staging overlaps main and is separate. GC is already inside phase durations. non_cpu_wall includes blocking, GIL and scheduling; CPU includes spin waits. Phase deltas are attribution, not a proven root cause or primary performance/gain estimate. No samples excluded.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('capture',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.write_text(json.dumps(analyze(a.capture),indent=2)+'\n')
