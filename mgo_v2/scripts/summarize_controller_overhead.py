#!/usr/bin/env python3
"""Publish the reduced, single-sample controller comparison with strict parity."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from run_controller_overhead import ROOT,OUT,ORIGINAL


def read(path):return json.loads(path.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def main():
    assert read(OUT/'cpu_differential_validation.json')['status']=='PASS'
    assert read(OUT/'gpu_profile_validation.json')['status']=='PASS'
    rows=[];checks=[];sources=[];safety=[]
    for variant in ('C0','C1','C2'):
        root=ROOT/f'control_{variant}';state=read(root/'status.json');assert state['status']=='PASS'
        for sample in state['memory']:safety.append(sample)
        for policy in ('P0','P1','O0'):
            name=f'b8_{policy}_{variant}'
            receipts=[read(root/f'receipts/{name}-rep0-rank{rank}.json') for rank in range(4)]
            evidence=[read(root/f'receipts/{name}-evidence-rank{rank}.json') for rank in range(4)]
            meta=[read(root/f'receipts/{name}-controller-rank{rank}.json') for rank in range(4)]
            for rank in range(4):
                original=read(Path('/home/hwlee/mgo-results/rank_demand_oracle_20261002')/f'timing_b8/receipts/b8_{policy}_rep0-evidence-rank{rank}.json')
                assert evidence[rank]['events']==original['events'] and evidence[rank]['tokens']==original['tokens']
                assert receipts[rank]['generated_token_ids']==evidence[rank]['tokens']
                assert len(evidence[rank]['events'])==3120
                assert evidence[rank]['events']==evidence[0]['events']
                assert receipts[rank]['metrics']==receipts[0]['metrics']
                c0=read(ROOT/f'control_C0/receipts/b8_{policy}_C0-rep0-rank{rank}.json')
                for key in ('metrics','host_fetch_bytes','fetch_modes','cache_stats'):
                    assert receipts[rank][key]==c0[key],(variant,policy,rank,key)
            m=receipts[0]['metrics'];generation=receipts[0]['global_max_generation_seconds']
            controller=max(r['controller_seconds'] for r in receipts)
            row=dict(batch=8,policy=policy,controller=variant,repeats=1,
                     generation_seconds=generation,tpot_seconds=receipts[0]['tpot_seconds'],
                     ttft_seconds=receipts[0]['ttft_seconds'],controller_seconds_max_rank=controller,
                     controller_fraction=controller/generation,
                     h2d_bytes=sum(r['host_fetch_bytes'] for r in receipts),fetches=m['fetches'],reloads=m['reloads'],
                     remote_pairs=m['remote_token_rank_pairs'],expert_rows=sum(sum(e['loads']) for e in evidence[0]['events']),
                     planner_payload_bytes_per_event=meta[0]['payload_bytes_per_event'],
                     planner_logical_payload_bytes=meta[0]['logical_payload_bytes'],
                     modeled_planner_peer_tx_bytes=meta[0]['modeled_planner_peer_tx_bytes'])
            assert row['h2d_bytes']==row['fetches']*9437184
            rows.append(row);checks.append(dict(controller=variant,policy=policy,ranks=4,events=3120,status='PASS',
                original_routes_decisions_cache_tokens_equal=True,full_generation_device_work_counters_equal=True))
        sources.extend(root.glob('receipts/*.json'));sources.append(root/'status.json')
    assert len(rows)==9
    for row in rows:
        base=next(r for r in rows if r['controller']=='C0' and r['policy']==row['policy'])
        for metric in ('controller_seconds_max_rank','generation_seconds','tpot_seconds'):
            row[metric+'_reduction']=1-row[metric]/base[metric]
    (OUT/'e2e_comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (OUT/'e2e_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    components=read(OUT/'cpu_component_summary.json')
    for policy in ('P0','P1','O0'):
        a=next(r for r in components if r['policy']==policy and r['controller']=='C0')
        b=next(r for r in components if r['policy']==policy and r['controller']=='C1')
        assert a['counts']['candidate_visits']==b['counts']['candidate_visits']
        assert a['counts']['choose_calls']==b['counts']['choose_calls']
    broadcast=[]
    for rank in range(4):
        meta=read(ROOT/f'profile_P1/receipts/b8_P1_C2_profile-controller-rank{rank}.json')
        times=defaultdict(int)
        for event in meta['transport_diagnostics']:
            for key,value in event['times_ns'].items():times[key]+=value
        broadcast.append(dict(rank=rank,profile_events=432,payload_bytes_per_event=meta['payload_bytes_per_event'],
                              logical_payload_bytes=meta['logical_payload_bytes'],
                              component_seconds={k:v/1e9 for k,v in times.items()}))
    (OUT/'planner_transport.json').write_text(json.dumps(broadcast,indent=2)+'\n')
    state=read(ROOT/'profile_P1/status.json');assert state['status']=='PASS';safety.extend(state['memory'])
    memory=dict(min_host_available_gib=min(s['host_available_bytes'] for s in safety)/2**30,
                max_tree_rss_gib=max(s.get('group_rss_bytes',0) for s in safety)/2**30,
                min_selected_gpu_free_mib=min(g['free_mib'] for s in safety for g in s['gpu'].values()),
                guard_stops=0)
    for p in ROOT.glob('*/status.json'):
        state=read(p)
        if 'guard_stop' in state:memory['guard_stops']+=1
    validation=dict(status='PASS',primary_generations=9,batch=8,repeats_per_policy_controller=1,
                    profile_generations=2,profile_forwards=9,expanded_batches=False,checks=checks,memory=memory,
                    cpu_differential=read(OUT/'cpu_differential_validation.json'),gpu_profile=read(OUT/'gpu_profile_validation.json'))
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    report=['# Controller overhead repair: reduced B8 confirmation','',
        'Status: PASS. This is a separate follow-up to the completed oracle packet. At the owner’s request, the matrix was reduced to one fresh fixed-work generation per policy/controller (9 total), full captured-route differential validation and two short posthoc GPU profiles. No B4/B16 controller expansion was run.','',
        '## Fresh matched physical measurements','',
        '| Policy | Controller | E2E s | TPOT s | Max-rank controller s | Controller / E2E | Controller reduction | TPOT reduction |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        report.append(f'| {row["policy"]} | {row["controller"]} | {row["generation_seconds"]:.3f} | {row["tpot_seconds"]:.4f} | {row["controller_seconds_max_rank"]:.3f} | {row["controller_fraction"]:.1%} | {row["controller_seconds_max_rank_reduction"]:+.1%} | {row["tpot_seconds_reduction"]:+.1%} |')
    report+=['','C0 is the original controller, C1 is local indexed/vectorized Coverage, C2 uses the C1 planner on rank 0 plus a compact decision broadcast. P0/P1/O0 preserve their original policy trajectories. C0 controls here are fresh and do not replace the primary oracle packet.','',
        '**Single-sample descriptive measurements:** each cell has one sample; no confidence interval, stable speedup estimate or inference about small differences is justified. C0, C1 and C2 ran in that order on a shared host. The reduced matrix does not counterbalance controller order. All cells perform one prefill plus 64 decode forwards, including outputs after EOS.','',
        '## What changed and why','',
        'C1 keeps the original controller flow and admission/substitution implementations. Its cache maintains numeric slot/key/last-use views and per-layer resident sets; successful placement/eviction updates Coverage counts incrementally. Gate history uses the original float64 accumulation and float32 rounding. Array midranks retain exact ties and the original score/last-use/expert-key tie-break. Full consistency scans are gated in the primary path and exercised offline. Candidate visits are unchanged; repeated Python object construction and scoring/sorting overhead is reduced.','',
        'C2 retains all-rank routing collection. Rank 0 computes the C1 plan and sends fixed-width integer decisions; followers reconstruct effective weights in the original summation order and apply the exact cache operations. There is no second independent residency authority, replication or migration. Follower random-generator state is unused; planner failover is not implemented.','',
        '## CPU mechanism diagnostics','',
        'These are instrumented single-process replays of each fresh C0 physical raw trace, not production speed measurements. Serialization, full state comparison and offline consistency checks occur outside component timers. C0 and C1 see exactly the same raw routes and decisions.','',
        '| Policy | Controller | Total CPU controller s | Coverage rank/victim s | Candidate visits | Cache key items materialized | Full resident-set builds |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in components:
        report.append(f'| {r["policy"]} | {r["controller"]} | {r["controller_seconds"]:.3f} | {r["component_seconds"].get("coverage_rank_and_victim",0):.3f} | {r["counts"].get("candidate_visits",0):,} | {r["counts"].get("cache_key_items",0):,} | {r["counts"].get("full_resident_set_materializations",0):,} |')
    report+=['','## Interpretation of the reduced comparison','',
        'The measured bottleneck is repeated Coverage victim evaluation, not the Hungarian solver. In the P0 trace alone, 68,131 victim choices inspect 30,646,150 candidates (about 450 per choice). C0 also materializes 31,392,509 cache-key items and 68,131 full resident sets. C1 keeps every candidate visit and victim decision while removing those repeated full constructions and using incremental Coverage counts and array ranking. The separate CPU component replay supports this mechanism independently of the physical timing samples.','',
        'All three physical C1 samples reduce controller time substantially. This is consistent with host planning contributing to the previous critical path. It does not establish a stable end-to-end speedup magnitude: the fresh C0/P1 controller times vary from 87.844 to 145.518 seconds across ranks, and its 205.113-second generation is much slower than the original P1 B8 control samples. Shared-host scheduling variation is visible in the baseline itself.','',
        'C2 has lower observed E2E than C1 for P0/P1 (70.359 to 61.714 seconds and 69.870 to 61.295 seconds), but O0 is nearly unchanged (74.667 to 73.800 seconds). P0 maximum-rank controller time is also essentially unchanged between C1 and C2. These single samples do not establish a universal advantage for single-planner broadcast. Keep C1 and C2 separately selectable; do not infer that removing four concurrent copies of planning should provide a fourfold latency gain.','',
        'This overhead-only study supplies no new stable ranking of P0/P1/O0 placement policies. The original oracle packet remains the placement evidence. No additional repetitions or B4/B16 controller expansion are needed under the owner-reduced confirmation scope.']
    report+=['','Detailed exclusive history, substitution, admission, Coverage synchronization and cache components are retained in `cpu_components_*.csv` and `cpu_component_summary.json`.','',
        '## Planner transport and device work','',
        f'C2 broadcasts {next(r for r in rows if r["controller"]=="C2")["planner_payload_bytes_per_event"]:,} payload bytes per layer event. The logical payload and modeled planner-to-peer bytes are recorded separately from expert H2D traffic; they exclude NCCL protocol overhead and are not measured wire bytes.','',
        'The short P1 profiles cover one prefill plus eight decode forwards after primary timing. They compare C1/C2 against the same prefix of the original C0 profile. Per-event/rank expert rows, GEMM counts and physical expert-fetch bytes matched exactly. Full 65-forward runtime fetch/cache/route counters also match for all three policies. The short profile does not characterize late-decode timing or all policies.','',
        '| Rank | Planner compute s | Encode s | Broadcast/copies/wait s | Apply s |',
        '|---:|---:|---:|---:|---:|']
    for r in broadcast:
        t=r['component_seconds'];report.append(f'| {r["rank"]} | {t.get("planner_compute",0):.4f} | {t.get("plan_encode",0):.4f} | {t.get("planner_broadcast",0):.4f} | {t.get("plan_apply",0):.4f} |')
    report+=['','Follower broadcast time includes waiting for rank 0 to finish planning. It is not a pure network-latency estimate. CPU transport spans include required copies and synchronization; actual NCCL GPU intervals and payload-copy reconciliation are retained in the profile artifacts.','',
        'GPU durations below sum the maximum rank interval-union duration at each decode layer event over the eight profiled decode forwards. NCCL includes the added C2 planner broadcast; it also includes GPU waiting and is not pure communication service time.','',
        '| Controller | Expert GPU ms | GEMM ms | NCCL ms | Expert H2D ms |',
        '|---|---:|---:|---:|---:|']
    for r in read(OUT/'gpu_profile_summary.json'):
        prefix='sum_event_max_rank_'
        report.append('| '+r['controller']+' | '+' | '.join(f'{r[prefix+k+"_gpu_union_ms"]:.3f}' for k in ('expert','gemm','nccl','h2d'))+' |')
    report+=['',
        '## Correctness and resources','',
        'All 39 CPU tests passed (the 32 existing tests and seven optimization/codec tests); see `cpu_tests.txt`.','',
        'All 9,360 captured CPU events match complete C0/C1/follower plans, effective-route weights, cache owners/slots/timestamps and rolling history. All nine physical generations match the original packet event hashes and full generated tokens on every rank. Malformed payloads, rounding/ties and diagnostics parity are tested separately.','',
        f'Only physical GPUs 0,1,4,5 were used, one job at a time. Minimum host availability was {memory["min_host_available_gib"]:.1f} GiB; maximum process-tree RSS was {memory["max_tree_rss_gib"]:.1f} GiB; minimum selected-GPU free memory was {memory["min_selected_gpu_free_mib"]:,} MiB. Guard stops: {memory["guard_stops"]}. Shared-host interference remains a timing limitation.','',
        '## Artifacts and reproduction','',
        'Run `run_controller_overhead.py --stage C0`, `replay_controller_equivalence.py`, then stages C1, C2 and profiles, followed by `analyze_controller_profiles.py` and this summary script. Source/protocol hashes and the frozen C0 audit are in `measurement_manifest.json`. Raw receipts and traces remain under `/home/hwlee/mgo-results/controller_overhead_20261002`; large Nsight traces are not committed.','',
        'The optimized paths remain opt-in. This packet changes implementation overhead only; no placement objective, substitution rule or Coverage policy coefficient was tuned.']
    (OUT/'RESULTS.md').write_text('\n'.join(report)+'\n')
    sources.extend((ROOT/'profile_P1/receipts').glob('*.json'));sources.append(ROOT/'profile_P1/status.json')
    (OUT/'raw_receipts.json').write_text(json.dumps([dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(sources))],indent=2)+'\n')
    (OUT/'artifact_hashes.json').write_text(json.dumps({p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_hashes.json'},indent=2)+'\n')
    print(json.dumps({k:validation[k] for k in ('status','primary_generations','profile_generations','expanded_batches')}))

if __name__=='__main__':main()
