"""Explicit B3 repair decision; absence of timing cannot become a gain claim."""
import csv,json,hashlib
from pathlib import Path
from prepare_critical_microbench import ROOT,PACKET,write


def finalize():
    diagnostic_path=PACKET/'B3_H1b_DIAGNOSTIC_RESULTS.json'
    if not diagnostic_path.exists():diagnostic_path=PACKET/'B3_DIAGNOSTIC_RESULTS.json'
    diagnosis=json.loads(diagnostic_path.read_text());candidate=diagnosis.get('candidate','H1')
    correctness_path=PACKET/('B3_H1b_CORRECTNESS.json' if candidate=='H1b' else 'B3_CORRECTNESS.json')
    clean=ROOT/'b3/C30/B3_C30_CLEAN_B128_H64/result.json'
    timings=[];gate=dict(correctness=json.loads(correctness_path.read_text())['status']=='PASS',
                        diagnostic=diagnosis['status']=='DIAGNOSTIC_PASS')
    result=dict(candidate=candidate,status='DIAGNOSTIC_GATE_FAIL',repair_gate=gate,causal_case=None,c60_authorized=False,primary_timing_available=clean.exists())
    if clean.exists():
        timing=json.loads(clean.read_text());assert timing['status']=='PASS'
        for key,rows in timing['samples'].items():
            policy,mode=key.split('_')
            for i,row in enumerate(rows,1):timings.append(dict(cache='C30',policy=policy,executor=mode,repeat=i,**row))
        est={key:value['estimate'] for key,value in timing['gates'].items()}
        gate.update(stable=not timing['unstable'],br_tpot_within_one_percent=est['BR_'+candidate]['TPOT']<=1.01*est['BR_H0']['TPOT'])
        result['estimates_seconds']=est
        result['full_ranges_seconds']={key:g['range'] for key,g in timing['gates'].items()}
        result['repair_gains']={policy:{k:1-est[policy+'_'+candidate][k]/est[policy+'_H0'][k] for k in ('TPOT','E2E_wall')} for policy in ('BR','FCA')}
        before=est['FCA_H0']['TPOT']/est['BR_H0']['TPOT']-1
        after=est['FCA_'+candidate]['TPOT']/est['BR_'+candidate]['TPOT']-1
        result['fca_slowdown']=dict(H0=before,H1=after)
        result['status']='REPAIR_PASS' if all(gate.values()) else 'REPAIR_GATE_FAIL'
        if result['status']=='REPAIR_PASS' and before>0:
            # Case A also requires corresponding reduction in measured skew.
            rows=diagnosis['rows']
            skew={mode:next(r['return_gpu_start_spread_ms'] for r in rows if r['policy']=='FCA' and r['executor']==mode) for mode in ('H0',candidate)}
            if after<=.5*before and skew[candidate]<skew['H0']:result['causal_case']='A'
            elif abs(after-before)/before<.25:result['causal_case']='B'
            else:result['causal_case']='INTERMEDIATE_OR_SKEW_NOT_CONFIRMED'
            result['c60_authorized']=result['causal_case'] in ('A','B')
    else:
        result['timing_not_run_reason']='C30 clean timing is pending; correctness and diagnostic integrity must pass first.'
    result['no_oracle_authorized']=True
    write(PACKET/'B3_TIMING_REPEATS.json',dict(status='COMPLETE' if timings else 'PENDING_C30_TIMING',primary_timing=True,primary_timing_available=bool(timings),rows=timings,all_valid_samples_retained=True))
    with (PACKET/'B3_TIMING_REPEATS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['cache','policy','executor','repeat','E2E_wall','TPOT']);w.writeheader();w.writerows(timings)
    write(PACKET/'B3_REPAIR_DECISION.json',result)
    lines=['# B3 host executor repair', '', 'Status: '+result['status'], '',
           'R4 GPUs 0/1/4/5, C30/B128, frozen64, BF16 V3 P2/T2. Other users’ GPUs untouched.', '',
           f'H0 is unchanged; {candidate} replays the exact expert kernel against live slot weights.',
           'Full64 discovery/capture/validation occurs before any timing; no new graph entry or compile is permitted during measurement.', '',
           '| Policy | Host call reduction | Expert host-loop reduction | Counter/packet parity |',
           '|---|---:|---:|---|']
    for p,g in diagnosis['gates'].items():lines.append(f"| {p} | {g['host_call_reduction']:.2%} | {g['host_loop_reduction']:.2%} | {g['counters_and_packets_equal']} |")
    if candidate=='H1b':
        original=json.loads((PACKET/'B3_DIAGNOSTIC_RESULTS.json').read_text())
        lines+=['','Original H1 also failed the 40% full-loop gate:']
        for p,g in original['gates'].items():lines.append(f"- {p}: launch reduction {g['host_call_reduction']:.2%}, host-loop reduction {g['host_loop_reduction']:.2%}.")
    lines+=['','## Remaining host path','', '| Policy/executor | Loop | Kernel launch | Gather | Weight multiply | Ready selection incl. wait | Host ready wait | Slot-use record |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in diagnosis['rows']:
        lines.append(f"| {r['policy']}/{r['executor']} | {r['host_expert_loop_ms']:.2f} | {r['host_kernel_launch_ms']:.2f} | {r['host_gather_ms']:.2f} | {r['host_weight_ms']:.2f} | {r['host_ready_select_including_wait_ms']:.2f} | {r['host_ready_wait_ms']:.2f} | {r['host_record_use_ms']:.2f} |")
    lines+=['','Values are diagnostic mean-rank ms/decode step. Ready wait is nested inside ready selection: do not add both. Other host-loop time includes diagnostic bookkeeping. This does not isolate a unique GIL/driver/OS cause.', '', 'Removing expert invocation overhead exposes more readiness waiting. Gather/weighting and slot-use recording remain material; CUDA-graph replay alone does not establish the required full-loop repair. No grouped GEMM or scheduler change is authorized here.'] if not gate['diagnostic'] else []
    if timings:
        lines+=['','| Policy/executor | TPOT (s) | E2E (s) |','|---|---:|---:|']
        for key,e in result['estimates_seconds'].items():lines.append(f"| {key} | {e['TPOT']:.6f} | {e['E2E_wall']:.6f} |")
        lines+=['','All valid repeats and full ranges are retained in the accompanying JSON/CSV. Two stable repeats stop; at most one conditional third. No noise-based deletion.']
    else:lines+=['','No primary timing has been run. Instrumented capture wall times are not performance evidence.']
    lines+=['','C60 authorization: '+str(result['c60_authorized'])+'. Stage C oracle remains unauthorized.', '',
            'Nsight node tracing is separate from timing. H1 launch ranges include scratch copies and replay; host and GPU spans overlap and must not be added.',
            'Runtime repair is not an admission-method contribution. See graph signatures, correctness receipts and diagnostic source hashes.']
    (PACKET/'B3_RESULTS.md').write_text('\n'.join(lines)+'\n')
    sources={}
    for p in [PACKET/'STAGE_B3_HOST_EXECUTOR_REPAIR.md',diagnostic_path,correctness_path,PACKET/('B3_H1b_GRAPH_SIGNATURES.json' if candidate=='H1b' else 'B3_GRAPH_SIGNATURES.json')]:
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    if clean.exists():sources[str(clean)]=hashlib.sha256(clean.read_bytes()).hexdigest()
    captures=[]
    for row in diagnosis['rows']:
        cap=Path(row['capture']);captures.append(json.loads((cap/'status.json').read_text()))
    input_sources={}
    full=ROOT.parent/'policy_regime_20261005/C30/inputs_B128_H64'
    for name in ('receipt.json','input_receipt.json','BR_P2_proof.json','FCA_P2_proof.json','requests.json','teacher.npy','selected.npy','weights.npy','offsets.npy','gates.npy'):
        path=full/name
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        input_sources[str(path)]=h.hexdigest()
    write(PACKET/'B3_SOURCE_RECEIPTS.json',dict(sources=sources,diagnostic_sources=diagnosis['sources'],capture_statuses=captures,frozen_input_sha256=input_sources))
    return result
if __name__=='__main__':print(json.dumps(finalize(),indent=2))
