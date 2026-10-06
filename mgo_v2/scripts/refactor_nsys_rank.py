"""torchrun entrypoint: one independent Nsight report per physical rank."""
import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--case',required=True);p.add_argument('--output',required=True);a=p.parse_args()
rank=int(os.environ['RANK']);output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
worker=Path(__file__).resolve().parents[1]/'examples/refactor_profile_worker.py'
case=json.loads(Path(a.case).read_text())
cmd=[case.get('profile_nsys_binary','nsys'),'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','--output',str(output/f'profile_rank{rank}'),sys.executable,'-u',str(worker),'--inputs',a.inputs,'--case',a.case,'--output',a.output]
if case.get('profile_disable_device_event_trace',False):cmd.insert(2,'--cuda-event-trace=false')
if case.get('b3_executor') in ('H0','H1'):cmd.insert(2,'--cuda-graph-trace=node')
flush_ms=case.get('profile_cuda_flush_ms',0)
assert flush_ms in (0,600000)
cmd.insert(2,f'--cuda-flush-interval={flush_ms}')
os.execvp(cmd[0],cmd)
