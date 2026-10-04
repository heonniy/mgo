"""Read-only attribution snapshot from completed sample receipts, never live scans."""
import argparse,json,statistics
from pathlib import Path
from adaptive_timing import single_decision
ROOT=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004')
PACKET=Path(__file__).resolve().parents[1]/'experiments/decode_prefetch_runtime_refactoring_20261004'
def diagnose(out,case):
 records=[]
 for repeat in range(1,8):
  paths=[out/f'{case}_r{repeat}_measure_rank{r}.json' for r in range(8)]
  if not all(p.exists() for p in paths):break
  rows=[json.loads(p.read_text()) for p in paths]
  records.append(dict(repeat=repeat,E2E_wall=max(r['E2E_wall'] for r in rows),TPOT=max(r['TPOT'] for r in rows),copy_bytes=sum(r['scheduler_metrics']['bytes'] for r in rows),rank_scheduler=[r['scheduler_metrics'] for r in rows],rank_controller=[r['controller_counters'] for r in rows],rank_usage=[r['process_usage_delta'] for r in rows]))
 assert len(records)>=2
 def identical(key):return all(r[key]==records[0][key] for r in records[1:])
 state=json.loads((out/'status.json').read_text())
 result_path=out/f'{case}_result.json';gate=json.loads(result_path.read_text())['gate'] if result_path.exists() else single_decision(records)
 boundaries=[b for b in state.get('boundaries',[]) if b['key'].startswith(case+'_r')]
 return dict(status='DIAGNOSTIC_SNAPSHOT',case=case,completed_repeats=len(records),timing_gate=gate,copy_bytes_identical=identical('copy_bytes'),per_rank_logical_work_identical=identical('rank_controller'),per_rank_scheduler_identical=identical('rank_scheduler'),samples=records,boundaries=boundaries,causal_conclusion='Not established: identical logical/copy work excludes workload-count changes, but does not distinguish CPU scheduling, staging latency, NCCL waits, thermal or clock variation during the measured interval. Boundary snapshots are not continuous telemetry. No samples excluded.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--case',required=True);a=p.parse_args();r=diagnose(ROOT/a.label,a.case);path=PACKET/f'{a.label}_{a.case}_JITTER.json';path.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ('completed_repeats','copy_bytes_identical','per_rank_logical_work_identical','per_rank_scheduler_identical')}))
