"""Receipt-level B4 audit; never selects or removes timing observations."""
import json,hashlib
from pathlib import Path
from prepare_critical_microbench import ROOT,PACKET,P,write,sha
from physical_repeat_rule import decide,final_unstable

def main():
 corr=json.loads((PACKET/'B4_CORRECTNESS.json').read_text());assert corr['correctness_pass'] and len(corr['ranks'])==8
 for r in corr['ranks']:
  assert r['status']=='PASS' and r['reference']==r['observed'] and r['token_parity'] and not r['token_mismatches']
  assert r['executor']['graph_entries']==0 and r['executor']['persistent_workspace_bytes']==69206016
 clean=ROOT/'b4/C30/B4_C30_CLEAN_RETRY1_B128_H64';state=json.loads((clean/'status.json').read_text());assert state['status']=='PASS'
 assert state['group']['gpus']==[0,1,4,5]
 fp=state['common_stack']
 for name,digest in fp['code'].items():assert sha(P/name)==digest,('code changed during measurement',name)
 inputs=ROOT/'b4/C30/inputs_B128_H64'
 for name,digest in fp['inputs'].items():assert sha(inputs/name)==digest
 rows=[];files=[]
 for key,samples in state['result']['samples'].items():
  policy,mode=key.split('_');assert len(samples)==decide(samples[:2])['target_repeats']
  for i,sample in enumerate(samples,1):
   rs=[]
   for rank in range(4):
    path=clean/f'{key}_r{i}_measure_rank{rank}.json';r=json.loads(path.read_text());rs.append(r);files.append(path)
    warm=json.loads((clean/f'{policy}_{mode}_correctness_rank{rank}.json').read_text())
    assert r['no_compile_in_measure'] and r['status']=='PASS' and r['argmax_hash']==warm['argmax_hash'] and r['counters']==warm['reference']
   assert sample=={m:max(r[m] for r in rs) for m in ('TPOT','E2E_wall')}
   rows.extend(rs)
 order=['BR_H1b','FCA_H1b','BR_H2','FCA_H2']
 initial=[f'{key}_r1' for key in order]+[f'{key}_r2' for key in reversed(order)]
 observed=[b['key'] for b in state['boundaries']];assert observed[:8]==initial
 result=dict(status='PASS',physical_samples=len(rows)//4,rank_receipts=len(rows),all_valid_samples_retained=True,counterbalanced_initial_order_verified=True,runtime_and_input_hashes_unchanged=True,correctness_rank_receipts=8,standalone_h2_graph_entries=0,workspace_bytes_per_gpu=69206016,physical_gpus=[0,1,4,5],minimum_timing_host_available_bytes=min(b['sample']['host_available_bytes'] for b in state['boundaries']),peak_timing_gpu_allocated_bytes=max(r['peak_gpu_bytes'] for r in rows),peak_correctness_gpu_allocated_bytes=max(r['peak_gpu_bytes'] for r in corr['ranks']),source_hashes={str(p):sha(p) for p in files})
 write(PACKET/'B4_AUDIT.json',result);print('B4 audit PASS')
if __name__=='__main__':main()
