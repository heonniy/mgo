#!/usr/bin/env python3
"""Fixed C2 T0-only actual/self matrix, with owner-worker restoration."""
import json,os,signal,time
from pathlib import Path
import batch_comm_common as common
from run_hot_expert_microcost import apps,restore,TARGETS
from run_rank_oracle_study import write
from trace_comm_input import sha
P=Path(__file__).resolve().parents[1]/'experiments/synthetic_comm_price_crossover_20261003'
ROOT=Path('/home/hwlee/mgo-results/synthetic_comm_price_crossover_20261003/C2')

def main():
    freeze=json.loads((P/'frozen_rho_trace_summary.json').read_text());assert freeze['status']=='PASS'
    assert json.loads((P/'resource_price_sweep.json').read_text())['status']=='PASS'
    for r in freeze['schedules']:
        for kind in ('actual','self','schedule'):
            item=r[kind];assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256']
    assert not ROOT.exists(),'No repeated timing attempts'
    ROOT.mkdir();common.ROOT=ROOT;common.PACKET=P
    original=json.loads((common.LOAD/'processes.json').read_text())
    paused=[r for r in original if r['gpu'] in TARGETS and common.owned(r['pid'])]
    assert all(pid in {r['pid'] for r in paused} for gpu,pid in apps() if gpu in TARGETS),'Foreign target-GPU job'
    state=dict(status='RUNNING',workers_before=original,paused_workers=paused,started_unix=time.time(),cells=[],transport='T0 only')
    write(P/'C2.json',state)
    try:
        for w in paused:os.kill(w['pid'],signal.SIGTERM)
        deadline=time.monotonic()+20
        while any(common.owned(w['pid']) for w in paused) and time.monotonic()<deadline:time.sleep(.25)
        for w in paused:
            if common.owned(w['pid']):os.kill(w['pid'],signal.SIGKILL)
        deadline=time.monotonic()+20
        while any(g in TARGETS for g,_ in apps()) and time.monotonic()<deadline:time.sleep(.5)
        assert not any(g in TARGETS for g,_ in apps())
        common.run('smoke_T0','ipc_rebase_smoke.py',['--mode','T0'],mode='T0',smoke=True)
        state['smoke']='smoke_T0';write(P/'C2.json',state)
        for pass_id in (0,1):
            schedules=freeze['schedules'] if pass_id==0 else list(reversed(freeze['schedules']))
            for entry in schedules:
                rho=entry['rho'];label=f'pass{pass_id}_rho{rho}'
                common.run(label,'synthetic_price_comm.py',['--counts',entry['actual']['path'],'--self-counts',entry['self']['path'],'--pass-index',pass_id],mode='T0')
                state['cells'].append(dict(label=label,rho=rho,pass_index=pass_id));write(P/'C2.json',state)
        state['status']='PASS'
    except BaseException as exc:state.update(status='FAIL',error=repr(exc));raise
    finally:
        try:state['restored_workers']=restore(original,paused)
        except BaseException as exc:state.update(status='FAIL',restore_error=repr(exc));raise
        finally:
            state['finished_unix']=time.time();state['receipts']=[dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)) for f in ROOT.glob('*/*') if f.is_file()];write(P/'C2.json',state)
            print(json.dumps({k:state.get(k) for k in ('status','error','restore_error','restored_workers')}),flush=True)
if __name__=='__main__':main()
