#!/usr/bin/env python3
"""Fail-fast physical correctness audit before the slower CPU matrix completes."""
import hashlib
import json
from summarize_trajectory_study import ROOT,OUT,PACKAGE,read,events,POLICIES,EXPERT_BYTES

def main():
    state=read(ROOT/'status.json');assert state['status']=='PHYSICAL_COMPLETE' and state['exit_code']==0
    assert state['world']==4 and state['physical_gpus']==[0,1,4,5] and len(state['cells'])==6
    assert len(list((ROOT/'physical').glob('*rep0-rank*.json')))==24
    for name,expected in state['source_sha256'].items():
        assert hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest()==expected
    cells=[]
    for batch in (4,8,16):
        for policy in POLICIES:
            label=f'b{batch}_{policy}'
            reference=events(ROOT/'physical'/f'{label}-events-rank0.jsonl')
            assert len(reference)==3120
            fetches=0
            for rank in range(4):
                rows=events(ROOT/'physical'/f'{label}-events-rank{rank}.jsonl')
                physical=events(ROOT/'physical'/f'{label}-physical-rank{rank}.jsonl')
                receipt=read(ROOT/'physical'/f'{label}-rep0-rank{rank}.json')
                assert len(rows)==len(physical)==3120
                for index,(r,p,ref) in enumerate(zip(rows,physical,reference)):
                    assert r['event']==p['event']==index and r['layer']==index%48
                    assert r['plan_sha256']==ref['plan_sha256'] and r['cache_sha256']==ref['cache_sha256']
                    assert r['controller_ns']==sum(r['times_ns'].values())+r['timer_unattributed_ns']
                    assert p['fetches']==sum(a[2]==rank for a in r['admissions'])
                    assert p['physical_h2d_bytes']==p['fetches']*EXPERT_BYTES
                assert sum(p['physical_h2d_bytes'] for p in physical)==receipt['host_fetch_bytes']
                fetches+=sum(p['fetches'] for p in physical)
                assert receipt['status']=='PASS' and receipt['world']==4
                assert receipt['config']['admission']==policy and receipt['config']['global_cache_ratio']==.3
                assert receipt['config']['gate_window']==128 and receipt['config']['gate_protect_threshold']==.2 and receipt['config']['similarity_threshold']==.65
                assert all(len(t)==65 for t in receipt['generated_token_ids'])
                assert receipt['metrics']['fetches']==sum(r['miss'] for r in rows)
            assert fetches==sum(r['miss'] for r in reference)
            cells.append(dict(batch=batch,policy=policy,events=3120,fetches=fetches,status='PASS'))
    result=dict(status='PASS',cells=cells,rank_receipts=24,rank_events=74880,world=4,physical_gpus=[0,1,4,5],
        note='Per-event plan/cache agreement across ranks, native physical fetch reconciliation, frozen inputs/config/source and output lengths verified. Full final packet adds CPU replay and baseline-token comparisons.')
    (OUT/'physical_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
