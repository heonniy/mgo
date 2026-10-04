"""Run the frozen-schedule E2E matrix with CoSLoT-style transport."""
import argparse
from run_env_offload_cell import run

def main(a):
 for policy in a.policies:
  run(a.cell,policy,'COMPILE',a.environment,0,'coslot')
  for repeat in range(a.repeats):
   run(a.cell,policy,'MEASURE',a.environment,repeat,'coslot')
  run(a.cell,policy,'COUNTERS',a.environment,0,'coslot')

if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--cell',choices=['P','R','E'],default='R')
 p.add_argument('--environment',choices=['env1','env2'],default='env1')
 p.add_argument('--policies',nargs='+',choices=['BR','CA','CA-rep'],default=['BR','CA'])
 p.add_argument('--repeats',type=int,default=3)
 main(p.parse_args())
