"""Freeze search subsets; proxy replay never supplies serving speedup claims."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from pcie_host import write


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_generation(path,repeat,n):
    ranks=[json.loads((path/f'repeat{repeat}_rank{r}.json').read_text()) for r in range(4)]
    trace=np.load(path/f'policy_trace_repeat{repeat}_rank0.npy')
    assert trace.shape==(n*48,61)
    for r,rank in enumerate(ranks):
        assert rank['finite_logits'] and rank['no_compile'] and rank['triton_no_compile']
        assert rank['validation']['status']=='PASS' and rank['validation']['physical_slots']==1843
        assert rank['validation']['scheduler']['background_copies']==0
        assert rank['expert_cache_start']=='empty' and rank['output_tokens']==n
        assert rank['metadata']=='native C++' and rank['expert_executor']=='grouped_decode_native_prefill'
        assert np.array_equal(trace,np.load(path/f'policy_trace_repeat{repeat}_rank{r}.npy'))
    return trace,ranks


def main(a):
    spec=json.loads(a.spec.read_text());assert spec['status']=='FROZEN'
    cohort=json.loads((a.root/'result.json').read_text());assert cohort['status']=='PASS'
    assert cohort['search_spec_sha256']==sha(a.spec)
    assert json.loads((a.root/'status.json').read_text())['status']=='PASS'
    a.out.mkdir(parents=True,exist_ok=False)
    rows=[]
    if a.stage=='nomination':
        import mgo_v2  # Initialize the runtime package before its policy adapter.
        from env_offload_policy import Policy
        assert cohort['search_stage']=='nomination'
        for candidate in spec['candidates']:
            path=a.root/candidate['candidate']/'G-NEAR';actual,ranks=verify_generation(path,1,16)
            capture=path/'current_routing.npz';receipt=json.loads(Path(str(capture)+'.json').read_text())
            assert sha(capture)==receipt['sha256']
            with np.load(capture) as stream:
                raw={name:stream[name] for name in stream.files}
                assert np.array_equal(raw['events'],np.arange(768))
                summaries={}
                for arm,mode in (('R-NEAR','rank_order'),('G-NEAR','group_balanced')):
                    p=Policy([459,459,459,458],np.zeros((48,128,128),np.float32),False,7,42,native_pcie=True,quota_mode=mode)
                    for event in raw['events']:
                        lo,hi=raw['offsets'][event:event+2]
                        p.apply(int(event),raw['selected'][lo:hi].astype(np.int64),raw['weights'][lo:hi],
                                raw['origins'][lo:hi].astype(np.int64),raw['gate'][event],None)
                    trace=p.native_pcie.trace();assert trace.shape==actual.shape
                    if arm=='G-NEAR':
                        np.testing.assert_array_equal(actual,trace)
                        assert hashlib.sha256(np.ascontiguousarray(p.slots).tobytes()).hexdigest()==ranks[0]['validation']['state_hash']
                    np.save(a.out/(candidate['candidate']+'__'+arm+'.npy'),trace)
                    counts=trace[48:,48:52].astype(np.int64);critical=counts.reshape(-1,2,2).sum(axis=2).max(axis=1)
                    summaries[arm]=dict(critical_group_fetch_sum=int(critical.sum()),total_fetches=int(counts.sum()),
                         mean_m=float(counts.sum(axis=1).mean()),m_mod4_2_fraction=float(np.mean(counts.sum(axis=1)%4==2)))
                r=summaries['R-NEAR']['critical_group_fetch_sum'];g=summaries['G-NEAR']['critical_group_fetch_sum']
                rows.append(dict(candidate=candidate['candidate'],proxy_score=1-g/max(1,r),
                        replay=summaries,current_routing_sha256=receipt['sha256'],actual_G_replay_exact=True))
                print(json.dumps(rows[-1]),flush=True)
            # Native controller object buffers are reclaimed before next case.
        ordered=sorted(rows,key=lambda row:(-row['proxy_score'],row['candidate']))
        selected=[r['candidate'] for r in ordered[:8]]
        method='Descending 1 - sum(G critical-group fetches)/sum(R critical-group fetches); independent cold caches on common current G routing; decode only; label tie-break'
        assert len(selected)==8
    else:
        assert cohort['search_stage']=='screen'
        for candidate in spec['candidates']:
            pair={}
            for arm in ('R-NEAR','G-NEAR'):
                path=a.root/candidate['candidate']/arm;verify_generation(path,1,64)
                result=json.loads((path/'repeat1.json').read_text())
                assert not result['profiled'] and not result['nomination'] and result['status']=='PASS'
                pair[arm]=result['TPOT']
            rows.append(dict(candidate=candidate['candidate'],screen_tpot=pair,
                        screen_gain_percent=(1-pair['G-NEAR']/pair['R-NEAR'])*100))
        ordered=sorted(rows,key=lambda row:(-row['screen_gain_percent'],row['candidate']))
        selected=[r['candidate'] for r in ordered[:3]]
        method='Descending full64 unprofiled screen TPOT reduction; label tie-break; three finalists frozen BEFORE final repetitions'
        assert len(selected)==3
    index={c['candidate']:c for c in spec['candidates']}
    subset=dict(spec,candidates=[index[label] for label in selected],
                parent_spec=str(a.spec),parent_spec_sha256=sha(a.spec),selection_root=str(a.root),
                selection_method=method,selection_is_not_final_estimate=True)
    write(a.out/'SELECTION.json',dict(status='PASS',stage=a.stage,method=method,all_candidates=ordered,
          selected=selected,source_root=str(a.root),source_spec_sha256=sha(a.spec),
          scope='Nomination count proxy' if a.stage=='nomination' else 'Selection-only single physical R/G pair; final repeats still required'))
    write(a.out/'CANDIDATES.json',subset)
    print(json.dumps(dict(status='FROZEN',selected=selected,out=str(a.out))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('nomination','screen'),required=True)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
