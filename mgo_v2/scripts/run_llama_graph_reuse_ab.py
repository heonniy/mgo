"""Single-cell ordinary llama graph-reuse check with CUDA Graphs disabled.

Fixed condition: R4/C30/local-B32/input512/output64, CPU32/32.
Reference arm is the already measured CUDA-OFF / graph-reuse-ON run from
9917b9c. This script runs only the new BOTH-OFF arm and compares it against the
preserved reference to avoid unnecessary GPU work.
"""
import json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parents[1]
PACK=P/'experiments/main_table_global_workload_20261006/expanded_matrix'
ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')
TOOLS=Path('/home/hwlee/mgo-tools/headline-r4')
PY='/home/hwlee/sub-moe/phase01/.venv/bin/python'
BASE=str(TOOLS/'base-env/bin/python')
CELL='R4_C30_B32_L512_O64';THREADS=32;REPEATS=2
REF=ROOT/f'llama_cuda_graph_ab_{CELL}_off_v1'
NEW=ROOT/f'llama_both_graphs_off_{CELL}_v1'
def mean(x):return sum(x)/len(x)
def main():
 assert (REF/'status.json').exists(),'missing preserved CUDA-OFF/reuse-ON reference from 9917b9c'
 ref_status=json.loads((REF/'status.json').read_text());assert ref_status['status']=='PASS'
 ref_cfg=json.loads((REF/'config.json').read_text())
 assert ref_cfg['cuda_graphs_runtime']=='off' and ref_cfg['llama_graph_reuse'] is True
 if not (NEW/'status.json').exists():
  cmd=[PY,str(P/'scripts/run_headline_job.py'),
       '--label',NEW.name,'--system','llama.cpp-sync-both-graphs-off',
       '--worker','headline_llama_sync_worker.py','--cell',CELL,
       '--python',BASE,'--ranks','1','--repeats',str(REPEATS),'--timeout','28800',
       '--workloads',str(PACK/'WORKLOADS.json'),
       '--llama-threads',str(THREADS),'--llama-cuda-graphs','off','--llama-graph-reuse','off','--llama-expert-placement','legacy_tail']
  subprocess.run(cmd,check=True)
 status=json.loads((NEW/'status.json').read_text());assert status['status']=='PASS'
 new_cfg=json.loads((NEW/'config.json').read_text())
 assert new_cfg['cuda_graphs_runtime']=='off' and new_cfg['llama_graph_reuse'] is False
 assert ref_cfg['build']['binary_sha256']==new_cfg['build']['binary_sha256'],'A/B must use same binary'
 ref=[json.loads((REF/f'repeat{r}.json').read_text()) for r in range(1,REPEATS+1)]
 new=[json.loads((NEW/f'repeat{r}.json').read_text()) for r in range(1,REPEATS+1)]
 for i in range(REPEATS):assert ref[i]['tokens']==new[i]['tokens'],f'token mismatch repeat {i+1}'
 metrics={}
 for k in ('TTFT','TPOT','E2E'):
  a=[x[k] for x in ref];b=[x[k] for x in new];ma,mb=mean(a),mean(b)
  metrics[k]=dict(cuda_off_reuse_on=a,both_off=b,reuse_on_mean=ma,both_off_mean=mb,
                  both_off_minus_reuse_on=mb-ma,slowdown_percent=(mb/ma-1)*100)
 summary=dict(status='PASS',cell=CELL,local_batch=32,global_requests=128,input_tokens=512,
              output_tokens=64,cache_percent=30,cpu_threads=32,repeats=REPEATS,
              cuda_graphs='OFF in both arms',reference_graph_reuse='ON',new_graph_reuse='OFF',
              same_binary=True,exact_token_parity=True,
              sole_intended_runtime_difference='LLAMA_GRAPH_REUSE_DISABLE',metrics=metrics)
 out=PACK/'LLAMA_GRAPH_REUSE_AB_B32_L512_SUMMARY.json'
 out.write_text(json.dumps(summary,indent=2)+'\n');print(out)
if __name__=='__main__':main()
