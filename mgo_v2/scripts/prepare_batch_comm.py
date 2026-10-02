"""Recover historical prompt order and freeze nested batch sources without GPU use."""
import hashlib,json
from pathlib import Path
from batch_comm_common import PACKET,ROOT,BATCHES,write,sha

def main():
 ROOT.mkdir(exist_ok=False)
 source=Path('/home/hwlee/mgo-results/runtime_validation_20261001/screen_workload.json')
 expected=json.loads((PACKET.parent/'rank_demand_oracle_20261002/measurement_manifest.json').read_text())
 # Historical checked-in workload hash, also used by the unchanged B8 capture script.
 assert sha(source)=='ccb19832a3d4aef654466bca871340f0c3d6a9124bc5df9c9aae74c429460733','BLOCKED_PROMPT_PARITY'
 prior=json.loads((PACKET/'exact_payload_capture_summary.json').read_text())
 for receipt in prior['raw_receipts']:assert sha(receipt['path'])==receipt['sha256']
 capture=prior['sources']['capture_worker.py'];assert sha(capture['path'])==capture['sha256']
 assert "[rank*8:(rank+1)*8]" in Path(capture['path']).read_text()
 rows=json.loads(source.read_text()); batches={}
 for batch in BATCHES:
  ranks=[]
  for rank in range(4):
   indices=[block*32+rank*8+i for block in range(4) for i in range(8)][:batch]
   assert len(indices)==batch
   ranks.append([dict(workload_index=i,sample_id=rows[i]['sample_id'],question_sha256=hashlib.sha256(rows[i]['question'].encode()).hexdigest()) for i in indices])
  batches[str(batch)]=dict(local_batch=batch,global_batch=4*batch,ranks=ranks)
 for rank in range(4):
  assert batches['4']['ranks'][rank]==batches['8']['ranks'][rank][:4]
  assert batches['8']['ranks'][rank]==batches['16']['ranks'][rank][:8]
  assert batches['16']['ranks'][rank]==batches['32']['ranks'][rank][:16]
 assert len({r['workload_index'] for rank in batches['32']['ranks'] for r in rank})==128
 write(PACKET/'batch_comm_prompt_provenance.json',dict(status='PASS',workload_path=str(source),workload_sha256=sha(source),
  historical_manifest='rank_demand_oracle_20261002/measurement_manifest.json',historical_capture_script=capture,
  B8_source_receipts=prior['raw_receipts'],B8_reused=True,batches=batches,
  ordering='Rank-local B8 preserved; append successive 32-prompt blocks, 8 prompts per rank; smaller batches are rank-local prefixes.'))
 frozen=json.loads((PACKET/'ipc_trace_comm_replay.json').read_text())['cpu_result_sha256']
 assert all(sha(PACKET/n)==h for n,h in frozen.items())
 write(PACKET/'batch_comm_progress.json',dict(status='PREPARED',plan_commit='9ec43e83da94060b0a06c147518a2bb26a34fd73',
  user_amendment='Include local B32/global B128; run eight-GPU burn whenever experiments are not running.',batches=list(BATCHES),
  new_capture_batches=[4,16,32],completed=[],cpu_result_sha256=frozen))
 print('Prompt parity and nested B4/B8/B16/B32 provenance: PASS')
if __name__=='__main__':main()
