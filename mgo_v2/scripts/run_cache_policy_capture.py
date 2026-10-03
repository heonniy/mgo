#!/usr/bin/env python3
"""Exactly two guarded feature captures, preserving reference routing and tokens."""
import json,os,signal,time
from pathlib import Path
import batch_comm_common as common
from run_hot_expert_microcost import apps,restore,TARGETS
from run_rank_oracle_study import write
from trace_comm_input import sha
P=Path(__file__).resolve().parents[1]/'experiments/cache_eviction_substitution_20261003'
ROOT=Path('/home/hwlee/mgo-results/cache_eviction_substitution_20261003/captures')
OLD=Path('/home/hwlee/mgo-results/fetch_comm_pareto_p2p_20261002')
REFERENCES={8:OLD/'exact_payload_20261003/capture',32:OLD/'batch_comm_20261003/capture32'}
PROMPTS=P.parent/'fetch_comm_pareto_p2p_20261002/batch_comm_prompt_provenance.json'
def audit():
    rows=[]
    for batch,root in REFERENCES.items():
        for rank in range(4):
            path=root/f'rank{rank}.json';r=json.loads(path.read_text())
            assert r['status']=='PASS' and len(r['events'])==432
            assert all('gate_scores' not in e and 'full_router_probs' not in e for e in r['events'])
            rows.append(dict(batch=batch,rank=rank,path=str(path),bytes=path.stat().st_size,sha256=sha(path),events=432,gate_history_present=False))
    sim=Path('/home/hwlee/mgo-results/runtime_validation_20261001/similarity.npy')
    write(P/'trace_readiness.json',dict(status='NEEDS_TWO_AUTHORIZED_CAPTURES',references=rows,similarity=dict(path=str(sim),sha256=sha(sim)),prompt_manifest_sha256=sha(PROMPTS)))
def main():
    assert not ROOT.exists(),'No repeated model capture attempts'
    ROOT.mkdir(parents=True);common.ROOT=ROOT;common.PACKET=P
    original=json.loads((common.LOAD/'processes.json').read_text())
    paused=[r for r in original if r['gpu'] in TARGETS and common.owned(r['pid'])]
    assert all(pid in {r['pid'] for r in paused} for g,pid in apps() if g in TARGETS),'Foreign target-GPU job'
    state=dict(status='RUNNING',workers_before=original,paused_workers=paused,started_unix=time.time(),cells=[])
    write(P/'S0_capture.json',state)
    try:
        for w in paused:os.kill(w['pid'],signal.SIGTERM)
        deadline=time.monotonic()+20
        while any(common.owned(w['pid']) for w in paused) and time.monotonic()<deadline:time.sleep(.25)
        for w in paused:
            if common.owned(w['pid']):os.kill(w['pid'],signal.SIGKILL)
        deadline=time.monotonic()+20
        while any(g in TARGETS for g,_ in apps()) and time.monotonic()<deadline:time.sleep(.5)
        assert not any(g in TARGETS for g,_ in apps())
        for batch in (8,32):
            rows=common.run(f'B{batch}','cache_policy_capture.py',['--batch',batch,'--prompts',PROMPTS,'--reference',REFERENCES[batch]],model=True)
            state['cells'].append(dict(batch=batch,reference_parity=[r['reference_parity'] for r in rows]))
            write(P/'S0_capture.json',state)
        state['status']='PASS'
    except BaseException as exc:state.update(status='FAIL',error=repr(exc));raise
    finally:
        try:state['restored_workers']=restore(original,paused)
        except BaseException as exc:state.update(status='FAIL',restore_error=repr(exc));raise
        finally:
            state['finished_unix']=time.time();state['receipts']=[dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)) for f in ROOT.glob('*/*') if f.is_file()];write(P/'S0_capture.json',state)
            print(json.dumps({k:state.get(k) for k in ('status','error','restored_workers')}),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true');a=p.parse_args()
    audit() if a.audit else main()
