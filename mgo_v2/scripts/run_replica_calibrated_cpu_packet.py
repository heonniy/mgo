#!/usr/bin/env python3
"""Run Env1/Env2 microbench calibration, then the CPU policy packet."""
import argparse,subprocess
from pathlib import Path

P=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpus',default='0,1,2,3');a=p.parse_args()
 subprocess.run(['/home/hwlee/sub-moe/phase01/.venv/bin/python','-u',
                 str(P/'scripts/run_replica_phase_microbench.py'),'--gpus',a.gpus],cwd=P,check=True)
 subprocess.run(['/home/hwlee/sub-moe/phase01/.venv/bin/python','-u',
                 str(P/'scripts/run_r8_phase_aware_policy_cpu.py')],cwd=P,check=True)

if __name__=='__main__':main()
