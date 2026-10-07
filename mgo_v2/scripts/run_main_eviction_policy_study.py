#!/usr/bin/env python3
"""Capture B8/B16/B64 frozen routes once, then replay five MAIN eviction policies.

This script is intentionally bounded to R4/C30/input256/output64 and GPUs
0,1,4,5 through run_headline_job.py. It does not physically run five policy
variants; only three route-capture model jobs are needed.
"""
import copy,hashlib,json,subprocess,sys
from pathlib import Path

P=Path(__file__).resolve().parents[1]
PACK=P/'experiments/main_eviction_policy_20261007'
BASE=P/'experiments/main_table_global_workload_20261006/expanded_matrix'
HEADROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')
OUTROOT=Path('/home/hwlee/mgo-results/main_eviction_policy_20261007')
PY='/home/hwlee/sub-moe/phase01/.venv/bin/python'

def sha(p):
 return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def _slice_b8_manifest(src,dst):
 rows=json.loads(Path(src).read_text())['requests'];assert len(rows)==64
 picked=[]
 for r in range(4):picked.extend(rows[r*16:r*16+8])
 assert len(picked)==32
 Path(dst).write_text(json.dumps(dict(requests=picked),separators=(',',':'))+'\n')
 return sha(dst)

def build_workloads():
 OUTROOT.mkdir(parents=True,exist_ok=True);(OUTROOT/'manifests').mkdir(exist_ok=True)
 source=json.loads((BASE/'WORKLOADS.json').read_text())
 by={x['cell']:x for x in source['cells']}
 b16=copy.deepcopy(by['R4_C30_B16_L256_O64']);b64=copy.deepcopy(by['R4_C30_B64_L256_O64'])
 b8=copy.deepcopy(b16);b8.update(cell='R4_C30_B8_L256_O64',local_batch=8,global_requests=32)
 for phase in ('warmup','target'):
  dst=OUTROOT/'manifests'/f'R4_C30_B8_L256_O64_{phase}.json'
  digest=_slice_b8_manifest(b16[phase]['path'],dst)
  b8[phase]=dict(path=str(dst),sha256=digest)
 pack=dict(status='FROZEN_FOR_EVICTION_STUDY',physical_gpus=[0,1,4,5],
           scope='R4/C30/input256/output64; local B8/B16/B64',cells=[b8,b16,b64])
 path=OUTROOT/'WORKLOADS.json';path.write_text(json.dumps(pack,indent=2)+'\n');return path

def main():
 workloads=build_workloads();state=dict(status='RUNNING',captures=[],simulations=[])
 PACK.mkdir(exist_ok=True)
 for batch in (8,16,64):
  cell=f'R4_C30_B{batch}_L256_O64';label=f'main_eviction_trace_{cell}_v1';out=HEADROOT/label
  if not (out/'status.json').exists():
   cmd=[PY,str(P/'scripts/run_headline_job.py'),'--label',label,'--system','Ours-trace-only',
        '--worker','headline_ours_worker.py','--cell',cell,'--ranks','4','--repeats','1','--timeout','28800',
        '--workloads',str(workloads),'--policy','LA_CA_NEAR','--prefill-optimized','--prefill-layout-fast',
        '--decode-layout-fast','--record-main-eviction-trace']
   subprocess.run(cmd,check=True)
  status=json.loads((out/'status.json').read_text());assert status['status']=='PASS'
  trace=out/'main_eviction_trace.npz';meta=out/'main_eviction_trace_meta.json'
  assert trace.exists() and meta.exists()
  sim=PACK/f'B{batch}_SIMULATION.json'
  subprocess.run([sys.executable,str(P/'scripts/main_eviction_replay.py'),
                  '--trace',str(trace),'--meta',str(meta),'--output',str(sim)],check=True)
  state['captures'].append(dict(batch=batch,cell=cell,path=str(out),trace_sha256=sha(trace),meta_sha256=sha(meta)))
  state['simulations'].append(dict(batch=batch,path=str(sim),sha256=sha(sim)))
  (PACK/'RUN_STATE.json').write_text(json.dumps(state,indent=2)+'\n')
 state['status']='PASS';(PACK/'RUN_STATE.json').write_text(json.dumps(state,indent=2)+'\n')
 print(json.dumps(dict(status='PASS',batches=[8,16,64])))

if __name__=='__main__':main()
