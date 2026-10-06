"""Audit physical samples, repetition/order rules and frozen runtime fingerprints."""
import hashlib,json,math
from pathlib import Path
from prepare_critical_microbench import ROOT,PACKET,P,write
from physical_repeat_rule import decide,final_unstable

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def audit():
    out=ROOT/'b3/C30/B3_C30_CLEAN_B128_H64'
    status=json.loads((out/'status.json').read_text());result=json.loads((out/'result.json').read_text())
    assert status['status']==result['status']=='PASS'
    group=status['group'];assert group['gpus']==[0,1,4,5] and group['world']==4 and group['batch']==128 and group['horizon']==64
    assert group['environment']=='env1' and result['candidate']=='H1b'
    for case in group['cases']:
        assert case['P']==2 and case['trigger']=='T2' and case['partial_precision']=='bf16' and case['overlap']
    keys=['BR_H0','FCA_H0','BR_H1b','FCA_H1b'];assert list(result['samples'])==keys
    expected_order=[k+'_r1' for k in keys]+[k+'_r2' for k in reversed(keys)]
    hashes={};files=set()
    for key,rows in result['samples'].items():
        decision=decide(rows[:2]);assert len(rows)==decision['target_repeats']
        assert result['gates'][key]['initial']==decision
        assert result['gates'][key]['unstable']==final_unstable(rows)
        if len(rows)==3:expected_order.append(key+'_r3')
        policy,executor=key.split('_')
        for repeat,sample in enumerate(rows,1):
            ranks=[]
            for rank in range(4):
                path=out/f'{key}_r{repeat}_measure_rank{rank}.json';r=json.loads(path.read_text())
                ref=json.loads((out/f'{policy}_correctness_rank{rank}.json').read_text())
                assert r['status']=='PASS' and r['rank']==rank and r['repeat']==repeat
                assert r['policy']==policy and r['executor']==executor and r['no_compile_in_measure']
                assert r['argmax_hash']==ref['argmax_hash'] and r['counters']==ref['reference']==ref['observed']
                assert all(math.isfinite(r[k]) and r[k]>0 for k in ('TPOT','E2E_wall'))
                ranks.append(r);files.add(path.name);hashes[str(path)]=sha(path)
            for metric in ('TPOT','E2E_wall'):assert sample[metric]==max(r[metric] for r in ranks)
    assert files=={p.name for p in out.glob('*_measure_rank*.json')},'missing or unreported valid sample'
    assert [b['key'] for b in status['boundaries']]==expected_order
    identity=status['common_stack']
    for name,digest in identity['code'].items():assert sha(P/name)==digest,('runtime source changed',name)
    for name,digest in identity['inputs'].items():assert sha(out.parent/'inputs_B128_H64'/name)==digest,('frozen input changed',name)
    diagnosis=json.loads((PACKET/'B3_H1b_DIAGNOSTIC_RESULTS.json').read_text())
    assert diagnosis['diagnostic_integrity_pass']
    assert all(g['counters_and_packets_equal'] for g in diagnosis['gates'].values())
    write(PACKET/'B3_AUDIT.json',dict(status='PASS',physical_samples=sum(map(len,result['samples'].values())),rank_receipts=len(files),
          policies=['BR','FCA'],executors=['H0','H1b'],physical_gpus=[0,1,4,5],horizon=64,
          exact_token_and_counter_parity=True,no_compile_in_measure=True,all_valid_samples_retained=True,
          counterbalanced_order=expected_order,repeat_rule='Two; third only when >2% and <=5%; >5% unstable without extension.',
          runtime_and_input_fingerprints_unchanged=True,hashes=hashes,source_sha=status['source_sha']))
    return result
if __name__=='__main__':audit();print('B3 physical audit PASS')
