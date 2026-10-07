"""Single-cell CUDA Graph A/B for the revised llama.cpp CPU32 baseline.

Only R4/C30/local-B32/input512/output64 is permitted. Both arms use the same
CUDA-Graph-capable binary. The sole runtime difference is
GGML_CUDA_DISABLE_GRAPHS.
"""
import json,statistics,subprocess
from pathlib import Path
P=Path(__file__).resolve().parents[1]
PACK=P/'experiments/main_table_global_workload_20261006/expanded_matrix'
ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')
TOOLS=Path('/home/hwlee/mgo-tools/headline-r4')
PY='/home/hwlee/sub-moe/phase01/.venv/bin/python'
BASE=str(TOOLS/'base-env/bin/python')
CELL='R4_C30_B32_L512_O64'
THREADS=32
REPEATS=2

def mean(xs):return sum(xs)/len(xs)
def main():
 subprocess.run([PY,str(P/'scripts/build_headline_llama_sync.py')],check=True)
 build=json.loads((PACK/'LLAMA_BUILD.json').read_text())
 assert build['cmake_flags']['GGML_CUDA_GRAPHS:BOOL']=='ON' and build['cuda_graph_support'] is True
 outs={}
 for mode in ('on','off'):
  label=f'llama_cuda_graph_ab_{CELL}_{mode}_v1'
  out=ROOT/label
  if not (out/'status.json').exists():
   cmd=[PY,str(P/'scripts/run_headline_job.py'),
        '--label',label,'--system',f'llama.cpp-sync-cudagraph-{mode}',
        '--worker','headline_llama_sync_worker.py','--cell',CELL,
        '--python',BASE,'--ranks','1','--repeats',str(REPEATS),'--timeout','28800',
        '--workloads',str(PACK/'WORKLOADS.json'),
        '--llama-threads',str(THREADS),'--llama-cuda-graphs',mode,'--llama-graph-reuse','on']
   subprocess.run(cmd,check=True)
  status=json.loads((out/'status.json').read_text());assert status['status']=='PASS'
  cfg=json.loads((out/'config.json').read_text())
  assert cfg['cpu_threads']==THREADS and cfg['cpu_batch_threads']==THREADS
  assert cfg['cuda_graphs_runtime']==mode and cfg['llama_graph_reuse'] is True
  outs[mode]=out

 on=[json.loads((outs['on']/f'repeat{r}.json').read_text()) for r in range(1,REPEATS+1)]
 off=[json.loads((outs['off']/f'repeat{r}.json').read_text()) for r in range(1,REPEATS+1)]
 for r in range(REPEATS):
  assert on[r]['tokens']==off[r]['tokens'],f'ON/OFF token mismatch repeat {r+1}'
 metrics={}
 for key in ('TTFT','TPOT','E2E'):
  a=[x[key] for x in on];b=[x[key] for x in off]
  ma,mb=mean(a),mean(b)
  metrics[key]=dict(cuda_graph_on=a,cuda_graph_off=b,on_mean=ma,off_mean=mb,
                    off_minus_on=mb-ma,off_over_on=mb/ma,
                    slowdown_percent=(mb/ma-1)*100)
 summary=dict(status='PASS',cell=CELL,local_batch=32,global_requests=128,input_tokens=512,
              output_tokens=64,cache_percent=30,cpu_threads=32,repeats=REPEATS,
              exact_token_parity=True,same_binary=True,
              sole_intended_runtime_difference='GGML_CUDA_DISABLE_GRAPHS',
              llama_graph_reuse=True,metrics=metrics)
 out=PACK/'LLAMA_CUDA_GRAPH_AB_B32_L512_SUMMARY.json'
 out.write_text(json.dumps(summary,indent=2)+'\n');print(out)
if __name__=='__main__':main()
