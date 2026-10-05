"""Read-only attribution snapshot from completed sample receipts, never live scans."""
import argparse,json,statistics
from pathlib import Path
from adaptive_timing import single_decision
ROOT=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004')
PACKET=Path(__file__).resolve().parents[1]/'experiments/decode_prefetch_runtime_refactoring_20261004'
def diagnose(out,case):
 state=json.loads((out/'status.json').read_text());world=state.get('group',{}).get('world',8)
 records=[]
 for repeat in range(1,8):
  paths=[out/f'{case}_r{repeat}_measure_rank{r}.json' for r in range(world)]
  if not all(p.exists() for p in paths):break
  rows=[json.loads(p.read_text()) for p in paths]
  records.append(dict(repeat=repeat,E2E_wall=max(r['E2E_wall'] for r in rows),TPOT=max(r['TPOT'] for r in rows),copy_bytes=sum(r['scheduler_metrics']['bytes'] for r in rows),rank_scheduler=[r['scheduler_metrics'] for r in rows],rank_controller=[r['controller_counters'] for r in rows],rank_usage=[r['process_usage_delta'] for r in rows]))
 assert len(records)>=2
 def identical(key):return all(r[key]==records[0][key] for r in records[1:])
 result_path=out/'result.json'
 if result_path.exists():
  result=json.loads(result_path.read_text());gate=result['gate'];gate_scope='Authoritative completed paired BR/LA group, not a per-policy stopping decision'
 else:
  gate=None;gate_scope='No completed paired result; diagnostic records only, no repeat scheduling' 
 boundaries=[b for b in state.get('boundaries',[]) if b['key'].startswith(case+'_r')]
 submitted=[sum(s['copies'] for s in r['rank_scheduler']) for r in records]
 canceled=[sum(s['canceled'] for s in r['rank_scheduler']) for r in records]
 totals=[a+b for a,b in zip(submitted,canceled)]
 volume=[r['copy_bytes'] for r in records]
 conclusion=('Logical work and submitted-copy bytes are identical.' if identical('rank_controller') and identical('copy_bytes') else 'Logical work is identical, but actual submitted-copy volume differs; retain cancellation and scheduling differences.' if identical('rank_controller') else 'Logical work differs; investigate configuration or controller consistency before timing interpretation.')
 conclusion+=' Causal timing attribution remains unestablished. Boundary snapshots are not continuous telemetry. No samples excluded.'
 return dict(status='DIAGNOSTIC_SNAPSHOT',world=world,gate_scope=gate_scope,case=case,completed_repeats=len(records),timing_gate=gate,copy_bytes_identical=identical('copy_bytes'),per_rank_logical_work_identical=identical('rank_controller'),per_rank_scheduler_identical=identical('rank_scheduler'),samples=records,boundaries=boundaries,submitted_copies=submitted,canceled_copies=canceled,submitted_plus_canceled_identical=len(set(totals))==1,copy_bytes_relative_range=(max(volume)-min(volume))/statistics.mean(volume),causal_conclusion=conclusion)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--label',required=True);p.add_argument('--case',required=True);a=p.parse_args();r=diagnose(a.root/a.label,a.case);path=PACKET/f'{a.label}_{a.case}_JITTER.json';path.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ('completed_repeats','copy_bytes_identical','per_rank_logical_work_identical','per_rank_scheduler_identical')}))
