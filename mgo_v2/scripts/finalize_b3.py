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
                        diagnostic=diagnosis['status']=='DIAGNOSTIC_PASS',
                        diagnostic_integrity=diagnosis['diagnostic_integrity_pass'],
                        compiled_host_call_reduction_at_least70=all(g['host_call_reduction']>=.70 for g in diagnosis['gates'].values()),
                        full_expert_host_loop_reduction_at_least40=all(g['host_loop_reduction']>=.40 for g in diagnosis['gates'].values()),
                        no_extra_h2d_or_packet=all(g['counters_and_packets_equal'] for g in diagnosis['gates'].values()))
    result=dict(candidate=candidate,status='DIAGNOSTIC_GATE_FAIL',repair_gate=gate,causal_case=None,c60_authorized=False,primary_timing_available=clean.exists())
    if clean.exists():
        timing=json.loads(clean.read_text());assert timing['status']=='PASS'
        for key,rows in timing['samples'].items():
            policy,mode=key.split('_')
            for i,row in enumerate(rows,1):timings.append(dict(cache='C30',policy=policy,executor=mode,repeat=i,**row))
        est={key:value['estimate'] for key,value in timing['gates'].items()}
        gate.update(stable=not timing['unstable'],br_tpot_within_one_percent=est['BR_'+candidate]['TPOT']<=1.01*est['BR_H0']['TPOT'])
        result['estimates_seconds']=est
        result['timing_stability']=timing['gates']
        result['full_ranges_seconds']={key:g['range'] for key,g in timing['gates'].items()}
        result['repair_gains']={policy:{k:1-est[policy+'_'+candidate][k]/est[policy+'_H0'][k] for k in ('TPOT','E2E_wall')} for policy in ('BR','FCA')}
        before=est['FCA_H0']['TPOT']/est['BR_H0']['TPOT']-1
        after=est['FCA_'+candidate]['TPOT']/est['BR_'+candidate]['TPOT']-1
        result['fca_slowdown']=dict(H0=before,candidate=after,relative_reduction=(1-after/before) if before>0 else None)
        result['status']='REPAIR_PASS' if all(gate.values()) else 'REPAIR_GATE_FAIL'
        result['c30_physical_scope_complete']=True
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
           'R4 GPUs 0/1/4/5, C30/B128, frozen64, BF16 V3 P2/T2. Other users’ GPUs untouched.', 'Diagnostics use the fixed first8 decode-step prefix; clean primary E2E/TPOT uses all64 decode steps. Diagnostic means are not full64 timing estimates.', '',
           f'H0 is unchanged; {candidate} replays the exact expert kernel against live slot weights.',
           'Full64 discovery/capture/validation occurs before any timing; no new graph entry or compile is permitted during measurement.', '',
           '| Policy | Host call reduction | Expert host-loop reduction | Counter/packet parity |',
           '|---|---:|---:|---|']
    if timings:
        headline=f"Clean TPOT point-estimate reduction ({candidate} vs H0): BR {result['repair_gains']['BR']['TPOT']:.2%}, FCA {result['repair_gains']['FCA']['TPOT']:.2%}; timing stability passed: {gate['stable']}."
        lines[4:4]=[headline,'The declared full-loop repair threshold remains binding independently of any physical gain. C60 is gated on all repair criteria.','']
    for p,g in diagnosis['gates'].items():lines.append(f"| {p} | {g['host_call_reduction']:.2%} | {g['host_loop_reduction']:.2%} | {g['counters_and_packets_equal']} |")
    if candidate=='H1b':
        original=json.loads((PACKET/'B3_DIAGNOSTIC_RESULTS.json').read_text())
        lines+=['','Original H1 also failed the 40% full-loop gate:']
        for p,g in original['gates'].items():lines.append(f"- {p}: launch reduction {g['host_call_reduction']:.2%}, host-loop reduction {g['host_loop_reduction']:.2%}.")
    lines+=['','## Remaining host path','', '| Policy/executor | Loop | Kernel launch | Gather | Weight multiply | Ready selection incl. wait | Host ready wait | Slot-use record |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(diagnosis['rows'],key=lambda r:(r['policy'],r['executor'])):
        lines.append(f"| {r['policy']}/{r['executor']} | {r['host_expert_loop_ms']:.2f} | {r['host_kernel_launch_ms']:.2f} | {r['host_gather_ms']:.2f} | {r['host_weight_ms']:.2f} | {r['host_ready_select_including_wait_ms']:.2f} | {r['host_ready_wait_ms']:.2f} | {r['host_record_use_ms']:.2f} |")
    lines+=['','Values are diagnostic mean-rank ms/decode step. Ready wait is nested inside ready selection: do not add both. Other host-loop time includes diagnostic bookkeeping. This does not isolate a unique GIL/driver/OS cause.', '', 'Removing expert invocation overhead exposes more readiness waiting. Gather/weighting and slot-use recording remain material; CUDA-graph replay alone does not establish the required full-loop repair. No grouped GEMM or scheduler change is authorized here.'] if not gate['diagnostic'] else []
    if timings:
        lines+=['','| Policy/executor | TPOT (s) | E2E (s) |','|---|---:|---:|']
        for key,e in result['estimates_seconds'].items():lines.append(f"| {key} | {e['TPOT']:.6f} | {e['E2E_wall']:.6f} |")
        lines+=['',f"Observed TPOT reduction: BR {result['repair_gains']['BR']['TPOT']:.2%}; FCA {result['repair_gains']['FCA']['TPOT']:.2%}. Stability gate: {gate['stable']}. These are physical timing observations; the full repair gate still requires every host/correctness/stability criterion.",'','All valid repeats and full ranges are retained in the accompanying JSON/CSV. Two stable repeats stop; at most one conditional third. No noise-based deletion.']
    else:lines+=['','No primary timing has been run. Instrumented capture wall times are not performance evidence.']
    lines+=['','C60 authorization: '+str(result['c60_authorized'])+'. Stage C oracle remains unauthorized.', '',
            'Nsight node tracing is separate from timing. H1 launch ranges include scratch copies and replay; host and GPU spans overlap and must not be added.',
            'Runtime repair is not an admission-method contribution. See graph signatures, correctness receipts and diagnostic source hashes.']
    (PACKET/'B3_RESULTS.md').write_text('\n'.join(lines)+'\n')
    sources={}
    for p in [PACKET/'STAGE_B3_HOST_EXECUTOR_REPAIR.md',diagnostic_path,correctness_path,PACKET/('B3_H1b_GRAPH_SIGNATURES.json' if candidate=='H1b' else 'B3_GRAPH_SIGNATURES.json')]:
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    if clean.exists():
        sources[str(clean)]=hashlib.sha256(clean.read_bytes()).hexdigest()
        directory=clean.parent;portable=[]
        clean_status=json.loads((directory/'status.json').read_text());assert clean_status['status']=='PASS'
        sources[str(directory/'status.json')]=hashlib.sha256((directory/'status.json').read_bytes()).hexdigest()
        signature_packet=json.loads((PACKET/('B3_H1b_GRAPH_SIGNATURES.json' if candidate=='H1b' else 'B3_GRAPH_SIGNATURES.json')).read_text())
        for rank in range(4):
            source=directory/f'graph_signatures_rank{rank}.json';graph=json.loads(source.read_text())
            expected=set()
            for r in signature_packet['rows']:
                if r['rank']==rank:expected.update(map(tuple,r['signatures']))
            assert set(map(tuple,graph['signatures']))==expected, 'clean graph union differs from diagnostic discoveries'
            signatures=graph.pop('signatures')
            graph.update(rank=rank,signature_count=len(signatures),signature_sha256=hashlib.sha256(json.dumps(signatures,separators=(',',':')).encode()).hexdigest(),exact_union_of_committed_policy_signatures=True)
            portable.append(graph);sources[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
            for policy in ('BR','FCA'):
                path=directory/f'{policy}_correctness_rank{rank}.json'
                assert json.loads(path.read_text())['status']=='PASS'
                sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        write(PACKET/'B3_CLEAN_GRAPH_SIGNATURES.json',dict(status='PASS',rows=portable))
        lines+=['',f"Clean comparison graph cache: {min(r['signature_count'] for r in portable)}–{max(r['signature_count'] for r in portable)} exact signatures/rank; max persistent scratch {max(r['scratch_bytes'] for r in portable)/1024**3:.2f} GiB/rank. H0 retains the same allocated graph buffers but does not replay them.",'Graph discovery/capture is benchmark preparation, outside E2E/TPOT. Unseen signatures invalidate the run; this is not a deployable dynamic-shape executor claim.']
        (PACKET/'B3_RESULTS.md').write_text('\n'.join(lines)+'\n')
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
    resource=PACKET/'B3_RESOURCE_AUDIT.json'
    if resource.exists():
        r=json.loads(resource.read_text());assert r['status']=='PASS'
        lines+=['',f"Resource audit: peak worker GPU allocation {r['peak_worker_gpu_allocated_gib']:.2f} GiB; minimum recorded host available memory {r['min_recorded_primary_host_available_gib']:.2f} GiB. Primary boundary snapshots show no foreign GPU jobs. Only physical GPUs 0/1/4/5 were used; owned model inference workers were restored on those GPUs."]
        (PACKET/'B3_RESULTS.md').write_text('\n'.join(lines)+'\n')
    for name in ('B3_AUDIT.json','B3_RESOURCE_AUDIT.json','B3_RESULTS.md','B3_REPAIR_DECISION.json','B3_TIMING_REPEATS.csv','B3_TIMING_REPEATS.json','B3_CLEAN_GRAPH_SIGNATURES.json'):
        path=PACKET/name
        if path.exists():sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write(PACKET/'B3_SOURCE_RECEIPTS.json',dict(sources=sources,diagnostic_sources=diagnosis['sources'],capture_statuses=captures,frozen_input_sha256=input_sources))
    return result
if __name__=='__main__':print(json.dumps(finalize(),indent=2))
