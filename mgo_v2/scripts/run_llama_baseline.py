"""Selected C30 llama baseline: CPU32, both graphs OFF, balanced3 placement."""
import argparse,json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parents[1]
PACK=P/'experiments/main_table_global_workload_20261006/expanded_matrix'
ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')
PY='/home/hwlee/sub-moe/phase01/.venv/bin/python'
BASE='/home/hwlee/mgo-tools/headline-r4/base-env/bin/python'
CELLS=tuple(f'R4_C30_B{b}_L{l}_O64' for b in (16,32) for l in (256,512))
def command(a):
 if a.cell not in CELLS:raise ValueError('Only owner-authorized C30 B16/B32 input256/512 cells supported')
 if not a.label or Path(a.label).name!=a.label or a.label in ('.','..'):raise ValueError('label must be a single new directory name')
 if (ROOT/a.label).exists():raise FileExistsError('Preserve existing run; choose a new label')
 result=[PY,str(P/'scripts/run_headline_job.py'),'--label',a.label,'--system','llama.cpp-sync-balanced3-baseline','--worker','headline_llama_sync_worker.py','--cell',a.cell,'--python',BASE,'--ranks','1','--repeats',str(a.repeats),'--timeout','28800','--workloads',str(PACK/'WORKLOADS.json'),'--llama-threads','32','--llama-cuda-graphs','off','--llama-graph-reuse','off','--llama-expert-placement','balanced3']
 if a.smoke:result.append('--smoke')
 return result
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cell',choices=CELLS,required=True);parser.add_argument('--label',required=True);parser.add_argument('--repeats',type=int,choices=range(1,6),default=2);parser.add_argument('--smoke',action='store_true');parser.add_argument('--dry-run',action='store_true');a=parser.parse_args();cmd=command(a)
 if a.dry_run:print(json.dumps(cmd,indent=2));return
 subprocess.run(cmd,check=True)
if __name__=='__main__':main()
