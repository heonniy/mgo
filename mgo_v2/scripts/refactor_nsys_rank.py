"""torchrun entrypoint: one independent Nsight report per physical rank."""
import argparse,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--case',required=True);p.add_argument('--output',required=True);a=p.parse_args()
rank=int(os.environ['RANK']);output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
worker=Path(__file__).resolve().parents[1]/'examples/refactor_profile_worker.py'
cmd=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','--output',str(output/f'profile_rank{rank}'),sys.executable,'-u',str(worker),'--inputs',a.inputs,'--case',a.case,'--output',a.output]
os.execvp(cmd[0],cmd)
